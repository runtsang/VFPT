#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import random

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

class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x

class FNetBlock_hidden(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(x, dim=-1).real
    return x

class FNetBlock_sequence(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(x, dim=-2).real
    return x

class PromptedTransformer_Fourier(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer_Fourier, self).__init__(
            config, img_size, vis)
        
        self.prompt_config = prompt_config
        self.vit_config = config
        
        img_size = _pair(img_size)
        patch_size = _pair(config.patches["size"])

        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens

        if self.prompt_config.FOURIER_TYPE == "fixed_linear" or self.prompt_config.FOURIER_TYPE == "linear":
            self.FT = nn.Linear(config.hidden_size, config.hidden_size)
            nn.init.kaiming_normal_(self.FT.weight, a=0, mode='fan_out')
        elif self.prompt_config.FOURIER_DIMENSION == "sequence":
            self.FT = FNetBlock_sequence()
        elif self.prompt_config.FOURIER_DIMENSION == "hidden":
            self.FT = FNetBlock_hidden()
        elif self.prompt_config.FOURIER_DIMENSION == "all":
            self.FT = FNetBlock()
        
        if self.prompt_config.FOURIER_PERCENTAGE != 0.0:
            self.fourier_num_tokens = math.floor(self.prompt_config.NUM_TOKENS * self.prompt_config.FOURIER_PERCENTAGE)
        else:
            self.fourier_num_tokens = 0
        
        self.fourier_addition_num_tokens = self.prompt_config.FOURIER_ADDITION_NUM
        
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

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"]-1

                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
                
                if self.prompt_config.FOURIER_ADDITION:
                    self.deep_prompt_fourier_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.fourier_addition_num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_fourier_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
        # Wrap self.prompt_embeddings in ParameterWrapper to be able to register hooks
        # self.prompt_embeddings = ParameterWrapper(self.prompt_embeddings.weight)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        
        if self.prompt_config.FOURIER_FIRST_LAYER and self.prompt_config.FOURIER_PERCENTAGE != 0.0 and self.prompt_config.FOURIER_HALF != "later":
            deep_prompt_fourier_emb = self.prompt_dropout(self.prompt_proj(
                            self.prompt_embeddings[0][:self.fourier_num_tokens]).expand(B, -1, -1))

            if self.fourier_num_tokens ==  self.num_tokens:
                x = torch.cat((
                    x[:, :1, :],
                    self.FT(deep_prompt_fourier_emb),
                    x[:, 1:, :]
                ), dim=1)
            else:
                if self.prompt_config.FOURIER_LOCATION == 'prepend':
                    x = torch.cat((
                        x[:, :1, :],
                        self.FT(deep_prompt_fourier_emb),
                        self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[0][self.fourier_num_tokens:]).expand(B, -1, -1)),
                        x[:, 1:, :]
                    ), dim=1)
                elif self.prompt_config.FOURIER_LOCATION == 'random':
                    selected_indices = random.sample(range(self.num_tokens), self.fourier_num_tokens)
                    tmp = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).clone()
                    for index in selected_indices:
                        tmp[:, index:index+1, :] = self.FT(self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[0][index:index+1]).expand(B, -1, -1)))
                    x = torch.cat((
                        x[:, :1, :],
                        tmp,
                        x[:, 1:, :]
                    ), dim=1)
                else:
                    x = torch.cat((
                        x[:, :1, :],
                        self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[0][self.fourier_num_tokens:]).expand(B, -1, -1)),
                        self.FT(deep_prompt_fourier_emb),
                        x[:, 1:, :]
                    ), dim=1)
        else:
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

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
                if i <= self.deep_prompt_embeddings.shape[0]:
                    if self.prompt_config.FOURIER_PERCENTAGE == 0.0:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1+self.num_tokens):, :]
                        ), dim=1)
                    elif self.prompt_config.FOURIER_HALF == "former" and i>=6:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1+self.num_tokens):, :]
                        ), dim=1)
                    elif self.prompt_config.FOURIER_HALF == "later" and i<=5:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1+self.num_tokens):, :]
                        ), dim=1)
                    elif self.prompt_config.MIXED == True and i%2 != 0:
                        # print('here')
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1+self.num_tokens):, :]
                        ), dim=1)
                    elif not self.prompt_config.FOURIER_ADDITION:
                        if self.fourier_num_tokens !=  self.num_tokens:
                            deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                                self.deep_prompt_embeddings[i-1][self.fourier_num_tokens:]).expand(B, -1, -1))
                        
                        deep_prompt_fourier_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i-1][:self.fourier_num_tokens]).expand(B, -1, -1))

                        if self.fourier_num_tokens ==  self.num_tokens:
                            hidden_states = torch.cat((
                                hidden_states[:, :1, :],
                                self.FT(deep_prompt_fourier_emb),
                                hidden_states[:, (1+self.num_tokens):, :]
                            ), dim=1)
                        else:
                            if self.prompt_config.FOURIER_LOCATION == 'prepend':
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    self.FT(deep_prompt_fourier_emb),
                                    deep_prompt_emb,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                            elif self.prompt_config.FOURIER_LOCATION == 'random':
                                selected_indices = random.sample(range(self.num_tokens), self.fourier_num_tokens)
                                tmp = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1:i]).expand(B, -1, -1)).clone()
                                for index in selected_indices:
                                    tmp[:, index:index+1, :] = self.FT(self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1][index:index+1]).expand(B, -1, -1)))
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    tmp,
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                            else:
                                hidden_states = torch.cat((
                                    hidden_states[:, :1, :],
                                    deep_prompt_emb,
                                    self.FT(deep_prompt_fourier_emb),
                                    hidden_states[:, (1+self.num_tokens):, :]
                                ), dim=1)
                    else:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                        
                        deep_prompt_fourier_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_fourier_embeddings[i-1]).expand(B, -1, -1))

                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            self.FT(deep_prompt_fourier_emb),
                            deep_prompt_emb,
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


class PromptedVisionTransformer_Fourier(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer_Fourier, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer_Fourier(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x_copy = x.clone()
        x, attn_weights = self.transformer(x)
        
        if self.prompt_cfg.VIS == True:
            aw = attn_weights[11].cpu().detach().numpy().mean(axis=1)
            
            np.savez(f'/home/{self.prompt_cfg.VIS_JSON_FOURIER}.npz', aw)
            
        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights