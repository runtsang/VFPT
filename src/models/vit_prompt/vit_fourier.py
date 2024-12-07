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
            
            np.savez(f'./{self.prompt_cfg.VIS_JSON_FOURIER}.npz', aw)
            
        x = x[:, 0]

        logits = self.head(x)

        if not vis:
            return logits
        return logits, attn_weights
    
import numpy as np
import math
from scipy.linalg import dft
import matplotlib.pyplot as plt

def calc_spectrum(A, F):
    return F @ A @ F.T

def plot_spectrum(a, len_tokens):
    F = dft(len_tokens, scale='sqrtn')
    s = calc_spectrum(a, F)
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
            
    plt.savefig(path,dpi=500)

import torch
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch16_224
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def grid_show(to_shows, cols, pth):
    rows = 1
    cols = 2
    it = iter(to_shows)
    image, title = next(it)
    plt.imshow(image, cmap='YlGnBu')
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar()
    plt.savefig(pth)

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols, pth):
    to_shows = []
    att_map = att_map.squeeze()
    # for i in range(att_map.shape[0]):
    #     to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols, pth=pth)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6, pth=None):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    plt.savefig(pth)
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image

def vis_attn(x, attention_maps, pn, name):
    # image = Image.open('./assets/dogcat.jpg')
    
    
    total = attention_maps[11].size()[2]
    
    visualize_heads(attention_maps[11][9,:,:,:].cpu().detach().numpy(), cols=4, pth=f"./layer_ave12")
    fig = plt.figure(figsize=(8,6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.)

    ax = fig.add_subplot(1,1, 1, projection='3d')
    
    xbase_index = [0,8,9,10,11,12,13,14,15,16,17, 23,30,37,44,51]
    num_tokens = len(xbase_index)
    xdata = np.array([xbase_index for i in range(total)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(total)])
    
    # print(attention_maps[11].size())
    tmp = attention_maps[11][9,:,0,:].unsqueeze(1)
    # print(tmp.size())
    for i in [1,2,3,4,5,6,7,8,9,10,11,60,109,158,206]:
        tmp = torch.cat((tmp,
                        attention_maps[11][9,:,i,:].unsqueeze(1)
                            ), dim=1)
        # print(tmp.size())
    # print(tmp.size())
    zdata = tmp.cpu().detach().numpy().mean(axis=0).T
    ax.plot_wireframe(xdata, ydata, zdata, rstride=0, color="royalblue", linewidth=1)

    ax.set_title(name, fontsize=20, fontweight="bold", y=1.015)
    plt.savefig(f"./3d_attention")
    
    for i in range(10):
        tensor = x[i,:,:,:]
        tensor = tensor.cpu().clone()
        tensor = tensor.squeeze(0)
        tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        attention_map = torch.cat((
                                attention_maps[11][i,:,:1,pn+1:],
                                attention_maps[11][i,:,pn+1:,pn+1:]
                            ), dim=1)
        
        tmp = torch.cat((attention_maps[11][i,:,:1,:1],
                        attention_maps[11][i,:,pn+1:,:1]
                            ), dim=1)
        
        attention_map = torch.cat((
                                tmp,
                                attention_map
                            ), dim=2)
        
        visualize_grid_to_grid_with_cls(attention_map.cpu().detach().numpy().mean(axis=0), 0, image, pth=f"./img_{str(i)}")
    


    
    