#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")

# class ParameterWrapper(nn.Parameter):
#     def __init__(self, data):
#         super(ParameterWrapper, self).__init__(data)
        
#     def register_forward_hook(self, hook):
#         self._forward_hooks.clear()
#         handle = self._register_forward_hook(hook)
#         return handle

class PromptedTransformer_AttentionSink(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer_AttentionSink, self).__init__(
            config, img_size, vis)
        
        self.prompt_config = prompt_config
        self.vit_config = config
        
        # ==============================================
        # Attention sink initialization
        # ==============================================
        
        self.sink_layers = None
        self.each_sink_number = self.prompt_config.SINK_NUMBER // self.prompt_config.SINK_LAYERS
        self.deep_sink_layers = None
        if self.prompt_config.SINK_RESIDUAL:
            self.residual_blocks = self.prompt_config.SINK_NUMBER // self.prompt_config.SINK_RESIDUAL_INTERVAL
            self.residual_blocks_index = [1]
            id = 1
            for rb in range(self.residual_blocks - 1):
                id += self.prompt_config.SINK_RESIDUAL_INTERVAL
                self.residual_blocks_index.append(id)
            self.residual_first_block_index = [i+1 for i in range(self.prompt_config.SINK_RESIDUAL_INTERVAL)]
            
        
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            # first layer
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            # sink tokens
            self.prompt_attention_sink_embeddings = nn.Parameter(torch.zeros(
                1, self.prompt_config.SINK_NUMBER, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_attention_sink_embeddings.data, -val, val)
            
            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"]-1
                
                # left layers
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
                
                if self.prompt_config.SINK_RESIDUAL:
                    # deep sink tokens
                    self.deep_prompt_attention_sink_embeddings = nn.Parameter(torch.zeros(
                        self.residual_blocks - 1, self.prompt_config.SINK_NUMBER, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_attention_sink_embeddings.data, -val, val)
                elif self.prompt_config.SINK_LAYERS != 1:
                    # deep sink tokens
                    self.deep_prompt_attention_sink_embeddings = nn.Parameter(torch.zeros(
                        self.prompt_config.SINK_LAYERS - 1, self.prompt_config.SINK_NUMBER, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_attention_sink_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
        # Wrap self.prompt_embeddings in ParameterWrapper to be able to register hooks
        # self.prompt_embeddings = ParameterWrapper(self.prompt_embeddings.weight)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        
        self.sink_layers = self.prompt_dropout(self.prompt_proj(self.prompt_attention_sink_embeddings).expand(B, -1, -1))
        if self.prompt_config.SINK_RESIDUAL:
            self.deep_sink_layers = []
            for rb in range(self.residual_blocks - 1):
                self.deep_sink_layers.append(self.prompt_dropout(self.prompt_proj(self.deep_prompt_attention_sink_embeddings[rb]).expand(B, -1, -1)))
        elif self.prompt_config.SINK_LAYERS != 1:
            self.deep_sink_layers = []
            for sl in range(self.prompt_config.SINK_LAYERS-1):
                self.deep_sink_layers.append(self.prompt_dropout(self.prompt_proj(self.deep_prompt_attention_sink_embeddings[sl]).expand(B, -1, -1)))

        if self.prompt_config.SINK_LOCATION == "prepend":
            x = torch.cat((
                    x[:, :1, :],
                    self.sink_layers,
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    self.sink_layers,
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        if self.prompt_config.SINK_LAYERS != 1 and not self.prompt_config.SINK_RESIDUAL:
            self.sink_layers[:, :self.each_sink_number, :]
            for sl in range(self.prompt_config.SINK_LAYERS-1):
                self.sink_layers = torch.cat((
                    self.sink_layers,
                    self.deep_sink_layers[sl][:, :self.each_sink_number, :]
                ), dim=1)
                
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                # the layers that need to be added
                if i <= self.deep_prompt_embeddings.shape[0]:
                    
                    # current layer index
                    layer_index = i + 1
                    
                    if not self.prompt_config.SINK_RESIDUAL:
                        if layer_index <= self.prompt_config.SINK_LAYERS:
                            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                                self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                            if self.prompt_config.SINK_LOCATION == "prepend":
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    self.deep_sink_layers[layer_index-2],
                                    deep_prompt_emb,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                            else:
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    deep_prompt_emb,
                                    self.deep_sink_layers[layer_index-2],
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                        else:
                            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                                self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                            
                            if self.prompt_config.SINK_LOCATION == "prepend":
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    self.sink_layers,
                                    deep_prompt_emb,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                            else:
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    deep_prompt_emb,
                                    self.sink_layers,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                    else:
                        if layer_index in self.residual_first_block_index:
                            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                                self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                            if self.prompt_config.SINK_LOCATION == "prepend":
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    self.sink_layers,
                                    deep_prompt_emb,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                            else:
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    deep_prompt_emb,
                                    self.sink_layers,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                        else:
                            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                                self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                            
                            if self.prompt_config.SINK_LOCATION == "prepend":
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    self.deep_sink_layers[int(math.ceil(layer_index/self.prompt_config.SINK_RESIDUAL_INTERVAL)-2)],
                                    deep_prompt_emb,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                            else:
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    deep_prompt_emb,
                                    self.deep_sink_layers[int(math.ceil(layer_index/self.prompt_config.SINK_RESIDUAL_INTERVAL)-2)],
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)

                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights


class PromptedVisionTransformer_AttentionSink(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer_AttentionSink, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer_AttentionSink(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights
