![image](https://github.com/runtsang/VFPT/blob/master/imgs/title.png)

**Official implementation of NeurIPS 2024 "Visual Fourier Prompt Tuning"**

**Contact me:** [runjia.tech](https://runjia.tech/) | rz4545@rit.edu | runjia@msn.com

[Paper](https://arxiv.org/abs/2411.01327) | [Homepage](https://runjia.tech/vfpt_page/)

# 📣News

(👉Under construction! You can currently check command.txt for commands. There are several redundancies in the current version, and the commands/instructions are not perfectly ready for formal release. I will gradually update it! Please stay tuned.)

**2024/12/07:** Our code is publicly available now! Thank you for your attention and patience!
**2024/12/02:** Our official [homepage](https://runjia.tech/vfpt_page/) is available now (slides and video are also included)! Check it out to see more details.
**2024/11/14:** Our preliminary key code is now available on GitHub.

# ⚡CODE TO-GO

If you are just interested in the key implementation in our paper, you can simply take out this part of the code.

### Visual Fourier Prompts


![image](https://github.com/user-attachments/assets/bf9d6ee9-ba07-4080-93dc-6414e849164d)

```python
# Visual Prompts
x = torch.cat((	x[:, :1, :],
                prompt_dropout(prompt_proj(prompt_embeddings).expand(B, -1, -1)), 
                x[:, 1:, :]), dim=1)

# Visual Fourier Prompts (Fourier percentage equals 1.0)
x = torch.cat((	x[:, :1, :],
                torch.fft.fft(torch.fft.fft(
                prompt_dropout(prompt_proj(prompt_embeddings).expand(B, -1, -1)), 
                                            dim=-1),dim=-2).real,
                x[:, 1:, :]), dim=1)
```

Our code implementation is based on [VPT](https://github.com/KMnP/vpt). I have also included part of the ViT VFPT implementation code (originally located at [src/models/vit_prompt/vit_fourier.py](https://github.com/runtsang/VFPT/blob/master/src/models/vit_prompt/vit_fourier.py) in the main root directory [./vit_VFPT.py](https://github.com/runtsang/VFPT/blob/master/vit_VFPT.py) for your convenience.

### Study of the Optimization

Our code implementation is based on [loss-landscape](https://github.com/tomgoldstein/loss-landscape).

### Study of the Interpretability

For the heatmap, our code implementation is based on [gradcam](https://github.com/1Konny/gradcam_plus_plus-pytorch). The attention map is simply obtained from the attention layer and visualized using Matplotlib.

# 📰Poster

![image](https://github.com/runtsang/VFPT/blob/master/imgs/VFPT.png)

# ❗Thanks

The documentation below is copied and modified  from [VPT](https://github.com/KMnP/vpt). Thanks for their effort.

## Environment settings

See `env_setup.sh`

## Structure of the this repo (key files are marked with 👉):

- `src/configs`: handles config parameters for the experiments.

  * 👉 `src/config/config.py`: <u>main config setups for experiments and explanation for each of them. </u> 

- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 

  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).


- `src/engine`: main training and eval actions here.

- `src/models`: handles backbone archs and heads for different fine-tuning protocols 

  * 👉`src/models/vit_prompt`: <u>a folder contains the same backbones in `vit_backbones` folder,</u> specified for VPT. This folder should contain the same file names as those in  `vit_backbones`

  * 👉 `src/models/vit_models.py`: <u>main model for transformer-based models</u> ❗️Note❗️: Current version only support ViT, Swin and ViT with mae, moco-v3

  * `src/models/build_model.py`: main action here to utilize the config and build the model to train / eval.

- `src/solver`: optimization, losses and learning rate schedules.  
- `src/utils`: helper functions for io, loggings, training, visualizations. 
- 👉`train.py`: call this one for training and eval a model with a specified transfer type.
- 👉`tune_fgvc.py`: call this one for tuning learning rate and weight decay for a model with a specified transfer type. We used this script for FGVC tasks.
- 👉`tune_vtab.py`: call this one for tuning vtab tasks: use 800/200 split to find the best lr and wd, and use the best lr/wd for the final runs
- `launch.py`: contains functions used to launch the job.

## Experiments

### Key configs:

- 🔥VFPT related:
  - MODEL.PROMPT_FOURIER.FOURIER_DIMENSION: all, sequence or hidden
  - MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE: 0.0 to 1.0
  - MODEL.PROMPT_FOURIER.FOURIER_LOCATION: append, prepend or random
- VPT related:
  - MODEL.PROMPT.NUM_TOKENS: prompt length
  - MODEL.PROMPT.DEEP: deep or shallow prompt
- Fine-tuning method specification:
  - MODEL.TRANSFER_TYPE
- Vision backbones:
  - DATA.FEATURE: specify which representation to use
  - MODEL.TYPE: the general backbone type, e.g., "vit" or "swin"
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: learning rate for the experiment
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment
  - DATA.BATCH_SIZE
- Datasets related:
  - DATA.NAME
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - RUN_N_TIMES: ensure only run once in case for duplicated submision, not used during vtab runs
  - OUTPUT_DIR: output dir of the final model and logs
  - MODEL.SAVE_CKPT: if set to `True`, will save model ckpts and final output of both val and test set

### Datasets preperation:

See Table 8 in the Appendix for dataset details. 

- Fine-Grained Visual Classification tasks (FGVC): The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found here: [Dropbox](https://cornell.box.com/v/vptfgvcsplits), [Google Drive](https://drive.google.com/drive/folders/1mnvxTkYxmOr2W9QjcgS64UBpoJ4UmKaM?usp=sharing). 

  - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- [Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/) (VTAB): see [`VTAB_SETUP.md`](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md) for detailed instructions and tips.

### Pre-trained model preperation

Download and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT` (ConvNeXt-Base and ResNet50 would be automatically downloaded via the links in the code). Note that you also need to rename the downloaded ViT-B/16 ckpt from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`.

See Table 9 in the Appendix for more details about pre-trained backbones.


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
<td align="center"><tt>d9715d</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
<td align="center"><tt>8cad7c</tt></td>
</tr>
<tr><td align="left">Swin-B</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">link</a></td>
<td align="center"><tt>bf9cc1</tt></td>
</tr>
<tr><td align="left">ConvNeXt-Base</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://pytorch.org/vision/stable/models.html">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
</tbody></table>


### Examples for training and visualization

```bash
# Training of VFPT
python tune_vtab.py \
--train-type "prompt" \
--config-file ./configs/prompt/prompt_fourier/Natural/caltech101_forVPT.yaml  \
MODEL.PROMPT_FOURIER.DEEP "True" \
MODEL.PROMPT_FOURIER.NUM_TOKENS "10" \
MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
OUTPUT_DIR "./output/" \
DATA.BATCH_SIZE "64"

# Landscape Visulization of VFPT
python ./ls_plot_surface.py \
--lr {lr} \
--wd {wd} \
--train-type "prompt" \
--x=-1:1:51 \
--y=-1:1:51 \
--config-file {config}  \
MODEL.PROMPT_FOURIER.DEEP "True"  \
MODEL.PROMPT_FOURIER.DROPOUT "0.10"  \
MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "{percentage}" \
MODEL.PROMPT_FOURIER.NUM_TOKENS "{num}" \
DATA.BATCH_SIZE "{batch_size}" \
OUTPUT_DIR "{output_dir}"

# GradCam Visulization of VFPT
PORT=20000 python \
tune_vtab_AS.py \
--train-type "prompt" \
--config-file ./configs/prompt/prompt_fourier/Natural/caltech101_forVPT.yaml  \
MODEL.PROMPT_FOURIER.DEEP "True" \
MODEL.PROMPT_FOURIER.NUM_TOKENS "10" \
MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
OUTPUT_DIR "./attn" \
DATA.BATCH_SIZE "64" \
ATTRIBUTION_TYPE "general" \
ATTRIBUTION_INTEGRATED_METHOD "pytorch_gradcam"
```





## Citation

If you find our work helpful in your research, please cite it as:

```
@inproceedings{zeng2024visual,
  title={Visual Fourier Prompt Tuning},
  author={Zeng, Runjia and Han, Cheng and Wang, Qifan and Wu, Chunshu and Geng, Tong and Huang, Lifu and Wu, Ying Nian and Liu, Dongfang},
  booktitle={NeurIPS},
  year={2024}
}
```

## License

The majority of VFPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.
