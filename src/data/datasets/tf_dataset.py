#!/usr/bin/env python3

"""a dataset that handles output of tf.data: support datasets from VTAB"""
import functools
import tensorflow.compat.v1 as tf
import torch
import torch.utils.data
import numpy as np
import random
import collections
from collections import Counter
from torch import Tensor
import torch.nn as nn

from ..vtab_datasets import base
# pylint: disable=unused-import
from ..vtab_datasets import caltech
from ..vtab_datasets import cifar
from ..vtab_datasets import clevr
from ..vtab_datasets import diabetic_retinopathy
from ..vtab_datasets import dmlab
from ..vtab_datasets import dsprites
from ..vtab_datasets import dtd
from ..vtab_datasets import eurosat
from ..vtab_datasets import kitti
from ..vtab_datasets import oxford_flowers102
from ..vtab_datasets import oxford_iiit_pet
from ..vtab_datasets import patch_camelyon
from ..vtab_datasets import resisc45
from ..vtab_datasets import smallnorb
from ..vtab_datasets import sun397
from ..vtab_datasets import svhn
from ..vtab_datasets.registry import Registry

from ...utils import logging
logger = logging.get_logger("visual_prompt")
tf.config.experimental.set_visible_devices([], 'GPU')  # set tensorflow to not use gpu  # noqa
DATASETS = [
    'caltech101',
    'cifar(num_classes=100)',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'patch_camelyon',
    'sun397',
    'svhn',
    'resisc45',
    'eurosat',
    'dmlab',
    'kitti(task="closest_vehicle_distance")',
    'smallnorb(predicted_attribute="label_azimuth")',
    'smallnorb(predicted_attribute="label_elevation")',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)',
    'clevr(task="closest_object_distance")',
    'clevr(task="count_all")',
    'diabetic_retinopathy(config="btgraham-300")'
]
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return x

import os
import random
from PIL import Image
import numpy as np

# 定义一个函数，实现随机选择文件夹并读取其中一张照片的功能，返回结果为numpy数组
def random_read_image_as_numpy(directory):
    # 获取当前目录下的所有文件夹
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    # 随机选择一个文件夹
    random_folder = random.choice(folders)
    # 获取该文件夹内的所有文件
    files = [f for f in os.listdir(os.path.join(directory, random_folder)) if os.path.isfile(os.path.join(directory, random_folder, f))]
    # 随机选择一个文件（假设都是图片文件）
    random_file = random.choice(files)
    # 读取图片并转化为numpy数组
    image = Image.open(os.path.join(directory, random_folder, random_file))
    image = np.array(image)
    return image

class TFDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
            "trainval"
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME

        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.get_data(cfg, split)

    # # origin
    # def get_data(self, cfg, split):
    #     tf_data = build_tf_dataset(cfg, split)
    #     data_list = list(tf_data)  # a list of tuples
        
    #     self._image_tensor_list = [t[0].numpy().squeeze() for t in data_list]
        
    #     self._targets = [int(t[1].numpy()[0]) for t in data_list]
    #     self._class_ids = sorted(list(set(self._targets)))

    #     logger.info("Number of images: {}".format(len(self._image_tensor_list)))
    #     logger.info("Number of classes: {} / {}".format(
    #         len(self._class_ids), self.get_class_num()))

    #     del data_list
    #     del tf_data
    
    # updated version of get_data to enable even sampling
    def get_data(self, cfg, split):
        tf_data = build_tf_dataset(cfg, split)
        data_list = list(tf_data)
        
        # for test, we don't need to sample images from each class
        if cfg.DATA.EVEN_SEPARETE and split != "test":
            random.seed(0) # set seed
            class_id_to_images = collections.defaultdict(list)
            for t in data_list:
                image_tensor = t[0].numpy().squeeze()
                target = int(t[1].numpy()[0])
                class_id_to_images[target].append(image_tensor)
                
            self._image_tensor_list = []
            self._targets = []
            if split == "train":
                for class_id, images in class_id_to_images.items():
                    num_images = min(len(images), cfg.DATA.MAX_IMAGES_PER_CLASS_TRAIN)
                    sampled_images = random.sample(images, num_images)
                    self._image_tensor_list += sampled_images
                    self._targets += [class_id] * num_images
            elif split == "val":
                for class_id, images in class_id_to_images.items():
                    num_images = min(len(images), cfg.DATA.MAX_IMAGES_PER_CLASS_VAL)
                    sampled_images = random.sample(images, num_images)
                    self._image_tensor_list += sampled_images
                    self._targets += [class_id] * num_images
            elif split == "trainval":
                for class_id, images in class_id_to_images.items():
                    number_trainval = cfg.DATA.MAX_IMAGES_PER_CLASS_TRAIN + cfg.DATA.MAX_IMAGES_PER_CLASS_VAL
                    num_images = min(len(images), number_trainval)
                    sampled_images = random.sample(images, num_images)
                    self._image_tensor_list += sampled_images
                    self._targets += [class_id] * num_images
            else:
                raise ValueError(F"split {split} is not supported.")
                
            self._class_ids = sorted(list(set(self._targets)))
            print("self._targets", self._targets)

            logger.info("Number of images: {}".format(len(self._image_tensor_list)))
            logger.info("Number of classes: {} / {}".format(
                len(self._class_ids), self.get_class_num()))

            del data_list
            del tf_data
            
        else:
            tf_data = build_tf_dataset(cfg, split)
            data_list = list(tf_data)  # a list of tuples
            
            self._image_tensor_list = [t[0].numpy().squeeze() for t in data_list]
            
            self._targets = [int(t[1].numpy()[0]) for t in data_list]
            self._class_ids = sorted(list(set(self._targets)))

            logger.info("Number of images: {}".format(len(self._image_tensor_list)))
            logger.info("Number of classes: {} / {}".format(
                len(self._class_ids), self.get_class_num()))

            del data_list
            del tf_data

    def get_info(self):
        num_imgs = len(self._image_tensor_list)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # index = 0
        # Load the image
        label = self._targets[index]
        im = to_torch_imgs(
            self._image_tensor_list[index], self.img_mean, self.img_std)

        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label
            # "id": index
        }
        # print("=====================================================")
        # print(index)
        return sample

    def __len__(self):
        if self.cfg.MODEL.PROMPT_FOURIER.VIS:
            return 64
        else:
            return len(self._targets)

class TFDataset_Fourier(torch.utils.data.Dataset):
    def __init__(self, cfg, split, target_path, LB):
        assert split in {
            "train",
            "val",
            "test",
            "trainval"
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg.DATA.NAME)
        logger.info("4. Fourier input Constructing {} dataset {}...".format(
            cfg.DATA.NAME, split))

        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME
        self.target_path = target_path
        self.LB = LB

        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.get_data(cfg, split)

    # # origin
    # def get_data(self, cfg, split):
    #     tf_data = build_tf_dataset(cfg, split)
    #     data_list = list(tf_data)  # a list of tuples
        
    #     self._image_tensor_list = [t[0].numpy().squeeze() for t in data_list]
        
    #     self._targets = [int(t[1].numpy()[0]) for t in data_list]
    #     self._class_ids = sorted(list(set(self._targets)))

    #     logger.info("Number of images: {}".format(len(self._image_tensor_list)))
    #     logger.info("Number of classes: {} / {}".format(
    #         len(self._class_ids), self.get_class_num()))

    #     del data_list
    #     del tf_data
    
    # updated version of get_data to enable even sampling
    def get_data(self, cfg, split):
        tf_data = build_tf_dataset(cfg, split)
        data_list = list(tf_data)
        
        # for test, we don't need to sample images from each class
        if cfg.DATA.EVEN_SEPARETE and split != "test":
            random.seed(0) # set seed
            class_id_to_images = collections.defaultdict(list)
            for t in data_list:
                image_tensor = t[0].numpy().squeeze()
                target = int(t[1].numpy()[0])
                class_id_to_images[target].append(image_tensor)
                
            self._image_tensor_list = []
            self._targets = []
            if split == "train":
                for class_id, images in class_id_to_images.items():
                    num_images = min(len(images), cfg.DATA.MAX_IMAGES_PER_CLASS_TRAIN)
                    sampled_images = random.sample(images, num_images)
                    self._image_tensor_list += sampled_images
                    self._targets += [class_id] * num_images
            elif split == "val":
                for class_id, images in class_id_to_images.items():
                    num_images = min(len(images), cfg.DATA.MAX_IMAGES_PER_CLASS_VAL)
                    sampled_images = random.sample(images, num_images)
                    self._image_tensor_list += sampled_images
                    self._targets += [class_id] * num_images
            elif split == "trainval":
                for class_id, images in class_id_to_images.items():
                    number_trainval = cfg.DATA.MAX_IMAGES_PER_CLASS_TRAIN + cfg.DATA.MAX_IMAGES_PER_CLASS_VAL
                    num_images = min(len(images), number_trainval)
                    sampled_images = random.sample(images, num_images)
                    self._image_tensor_list += sampled_images
                    self._targets += [class_id] * num_images
            else:
                raise ValueError(F"split {split} is not supported.")
                
            self._class_ids = sorted(list(set(self._targets)))
            print("self._targets", self._targets)

            logger.info("Number of images: {}".format(len(self._image_tensor_list)))
            logger.info("Number of classes: {} / {}".format(
                len(self._class_ids), self.get_class_num()))

            del data_list
            del tf_data
            
        else:
            tf_data = build_tf_dataset(cfg, split)
            data_list = list(tf_data)  # a list of tuples
            
            self._image_tensor_list = [t[0].numpy().squeeze() for t in data_list]
            
            self._targets = [int(t[1].numpy()[0]) for t in data_list]
            self._class_ids = sorted(list(set(self._targets)))

            logger.info("Number of images: {}".format(len(self._image_tensor_list)))
            logger.info("Number of classes: {} / {}".format(
                len(self._class_ids), self.get_class_num()))

            del data_list
            del tf_data

    def get_info(self):
        num_imgs = len(self._image_tensor_list)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        label = self._targets[index]
        im = to_torch_imgs(
            self._image_tensor_list[index], self.img_mean, self.img_std)
        
        if self._split == "train":
            index = index
            target = random_read_image_as_numpy(self.target_path)
            target = to_torch_imgs(
                target, self.img_mean, self.img_std)
            src_in_trg = FDA_source_to_target( im, target, L=self.LB )
            im = src_in_trg
        else:
            index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
            # "id": index
        }
        return sample

    def __len__(self):
        return len(self._targets)



def preprocess_fn(data, size=224, input_range=(0.0, 1.0)):
    image = data["image"]
    image = tf.image.resize(image, [size, size])

    image = tf.cast(image, tf.float32) / 255.0
    image = image * (input_range[1] - input_range[0]) + input_range[0]

    data["image"] = image
    return data


def build_tf_dataset(cfg, mode):
    """
    Builds a tf data instance, then transform to a list of tensors and labels
    """

    if mode not in ["train", "val", "test", "trainval"]:
        raise ValueError("The input pipeline supports `train`, `val`, `test`."
                         "Provided mode is {}".format(mode))

    vtab_dataname = cfg.DATA.NAME.split("vtab-")[-1]
    data_dir = cfg.DATA.DATAPATH
    # print('3 build tf dataset', vtab_dataname)
    # print('4 build tf dataset', data_dir)
    
    if vtab_dataname in DATASETS:
        data_cls = Registry.lookup("data." + vtab_dataname)
        vtab_tf_dataloader = data_cls(data_dir=data_dir)
    else:
        raise ValueError("Unknown type for \"dataset\" field: {}".format(
            type(vtab_dataname)))

    split_name_dict = {
        "dataset_train_split_name": "train800",
        "dataset_val_split_name": "val200",
        "dataset_trainval_split_name": "train800val200",
        "dataset_test_split_name": "test",
    }

    def _dict_to_tuple(batch):
        return batch['image'], batch['label']
    
    return vtab_tf_dataloader.get_tf_data(
        batch_size=1,  # data_params["batch_size"],
        drop_remainder=False,
        split_name=split_name_dict[f"dataset_{mode}_split_name"],
        preprocess_fn=functools.partial(
            preprocess_fn,
            input_range=(0.0, 1.0),
            size=cfg.DATA.CROPSIZE,
            ),
        for_eval=mode != "train",  # handles shuffling
        shuffle_buffer_size=1000,
        prefetch=1,
        train_examples=None,
        epochs=1  # setting epochs to 1 make sure it returns one copy of the dataset
    ).map(_dict_to_tuple)  # return a PrefetchDataset object. (which does not have much documentation to go on)


def to_torch_imgs(img: np.ndarray, mean: Tensor, std: Tensor) -> Tensor:
    t_img: Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    t_img -= mean
    t_img /= std

    return t_img
