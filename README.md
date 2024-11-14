![image](https://github.com/user-attachments/assets/bf9d6ee9-ba07-4080-93dc-6414e849164d)

Official implementation of NeurIPS 2024 "Visual Fourier Prompt Tuning"

Paper: https://arxiv.org/abs/2411.01327

# 📣News

If you urgently need access to the code, please feel free to contact me (rz4545@rit.edu), and I will promptly provide you with the original version.

Thank you for your understanding and patience!

# ✨TO-GO Code

I apologize for the delayed update, as I have been quite busy with the ICLR rebuttal and ACL 2024 recently. It may still take some time for me to organize the code, but I will make it publicly available no later than the NeurIPS 2024 conference (December 10, 2024).

If you are just interested in the key implementation in our paper, you can simply take out this part of the code.

## Visual Fourier Prompts

```python
import torch
import torch.nn as nn
from torch.nn import Dropout

# for visual prompt tuning in one layer, we have
prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
prompt_dropout = Dropout(self.prompt_config.DROPOUT)
prompt_proj = nn.Identity()
prompt_emb = prompt_dropout(prompt_proj(prompt_embeddings).expand(B, -1, -1))
x = torch.cat((	x[:, :1, :],
                prompt_emb,
                x[:, 1:, :]), dim=1)

# for visual fourier prompts where the fourier percentage equals one, we have
FT = FNetBlock()
fourier_prompt_emb = FT(prompt_emb)
x = torch.cat((	x[:, :1, :],
                fourier_prompt_emb,
                x[:, 1:, :]), dim=1)

class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x
```

Our code implementation is based on VPT (https://github.com/KMnP/vpt). I have also included part of the ViT VFPT implementation code (originally located at `src/models/vit_prompt/vit_VFPT.py`) in the main root directory for your convenience.

## Study of the Optimization

I followed the code from [here](https://github.com/tomgoldstein/loss-landscape) and will organize it later.

## Study of the Interpretability

For the heatmap, I used the code [here](https://github.com/1Konny/gradcam_plus_plus-pytorch). The attention map is simply obtained from the attention layer and visualized using Matplotlib. I will organize it later.
