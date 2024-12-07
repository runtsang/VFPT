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

class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)
        
        self.prompt_config = prompt_config
        self.vit_config = config
        
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

        else:
            raise ValueError("Other initiation scheme is not supported")
        
        # Wrap self.prompt_embeddings in ParameterWrapper to be able to register hooks
        # self.prompt_embeddings = ParameterWrapper(self.prompt_embeddings.weight)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
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
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
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


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False
    ):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(
            model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)

        # import json
        # json_path = '/home/ch7858/rz/vpt/temporary/vit.json'
        # with open (json_path) as f:
        #     vit_list = json.load(f)
        
        # s1=plot_spectrum(attn_weights[3][0,0,:,:].cpu().detach().numpy())
        # s2=plot_spectrum(attn_weights[7][0,0,:,:].cpu().detach().numpy())
        # s3=plot_spectrum(attn_weights[11][0,0,:,:].cpu().detach().numpy())
        
        # if len(vit_list) < 101:
        #     vit_list.append([s1,s2,s3])
        
        # with open(json_path, 'w') as write_file:
        #     json.dump(vit_list, write_file)
        
        # plot_6x12(attn_weights, '/home/ch7858/rz/vpt/temporary/vit_100')
            
        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights
    
import numpy as np
import math
from scipy.linalg import dft
import matplotlib.pyplot as plt

len_tokens = 247

F = dft(len_tokens, scale='sqrtn')

def calc_spectrum(A):
    return F @ A @ F.T

def plot_spectrum(a):
    s = calc_spectrum(a)
    s = np.linalg.norm(s, ord=2, axis=1)
    s = np.concatenate([s[-math.floor(len_tokens/2):], s[0:1], s[1:math.floor(len_tokens/2)]], axis=0)
    return s

def plot_6x12(attn_weights, path):
    for r in range(1, 13):
        for c in range(1, 7):
            i = (r-1)*6 + c
            plt.subplot(12, 6, i)
            s = plot_spectrum(attn_weights[r-1][0,c-1,:,:].cpu().detach().numpy())
            plt.plot(s)
            if r != 12:
                plt.xticks([])
            plt.tick_params(labelsize=3)
            
    plt.savefig(path,dpi=1000)

def compute_similarity(x, y):
    s = torch.matmul(torch.transpose(x, -2, -1), y) # [bs, h, l, l]
    l1, l2 = torch.linalg.norm(x, ord=2, dim=-2), torch.linalg.norm(y, ord=2, dim=-2)
    l = l1[..., :, None] * l2[..., None, :] # [bs, h, l, l]
    N = s.shape[-1]
    s = torch.abs(s / l)
    return s.mean().item()