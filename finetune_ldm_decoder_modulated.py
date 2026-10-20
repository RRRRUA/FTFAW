# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch import optim
import webdataset as wds
import utils
import utils_img
import utils_model
import torch.utils.tensorboard as tb
import braceexpand
import lpips
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from src.loss.loss_provider import LossProvider
from src.ldm.models.diffusion.ddpm import LatentDiffusion
from watermark.model.watermark_net import WaterMarkGenNet, gen_null_watermark, ConvNextBase, WaterMarkGenNet_NoMask, WaterMarkGenNetOneNoMask, gen_null_watermark_bias
from watermark.model.resnet import ResNet50
from watermark.util import param_util
from watermark.losses import losses, ssim
from watermark.util.image_augmentation import TestDataAugmentation, TrainDataAugmentation, fill_image, TrainDataAugmentationAll, TrainDataAugmentationNull
from compressai import zoo
from watermark.watermark_attack.wmattack import DiffWMAttackerOnline, VAEWMAttackerOnline
from watermark.watermark_attack.regen_pipe import ReSDPipeline
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# device = torch.device('cpu')


class LinearIncrease():
    def __init__(self, orig_w, max_w, nums):
        super(LinearIncrease, self).__init__()
        self.orig_w = orig_w
        self.max_w = max_w
        self.nums = nums
        self.iw = (max_w - orig_w) / nums
        self.w = orig_w

    def update(self):
        if self.w < self.max_w:
           self.w = self.w + self.iw
        return self.w


# a0_i = LinearIncrease(orig_w=0.05, max_w=1, nums=1000)
# a1_i = LinearIncrease(orig_w=2, max_w=50, nums=1000)
a0_i = LinearIncrease(orig_w=0.1, max_w=1, nums=500)
a1_i = LinearIncrease(orig_w=5, max_w=50, nums=500)


def get_compress_model_list(model_type='bmshj2018_hyperprior', metirc='mse'):
    model_list = []
    if 'bmshj2018' in model_type or 'mbt2018' in model_type:
        max_quality = 8
    else:
        max_quality = 6
    for i in range(1, max_quality + 1):
        if metirc == 'mse':
            if model_type == 'bmshj2018_factorized':
                model_list.append(zoo.bmshj2018_factorized(quality=i, metric='mse', pretrained=True, progress=True))
            if model_type == 'bmshj2018_hyperprior':
                model_list.append(zoo.bmshj2018_hyperprior(quality=i, metric='mse', pretrained=True, progress=True))
            if model_type == 'cheng2020_anchor':
                model_list.append(zoo.cheng2020_anchor(quality=i, metric='mse', pretrained=True, progress=True))
            if model_type == 'cheng2020_attn':
                model_list.append(zoo.cheng2020_attn(quality=i, metric='mse', pretrained=True, progress=True))
            if model_type == 'mbt2018_mean':
                model_list.append(zoo.mbt2018_mean(quality=i, metric='mse', pretrained=True, progress=True))
            if model_type == 'mbt2018':
                model_list.append(zoo.mbt2018(quality=i, metric='mse', pretrained=True, progress=True))
        else:
            if model_type == 'bmshj2018_factorized':
                model_list.append(zoo.bmshj2018_factorized(quality=i, metric='ms-ssim', pretrained=True, progress=True))
            if model_type == 'bmshj2018_hyperprior':
                model_list.append(zoo.bmshj2018_hyperprior(quality=i, metric='ms-ssim', pretrained=True, progress=True))
            if model_type == 'cheng2020_anchor':
                model_list.append(zoo.cheng2020_anchor(quality=i, metric='ms-ssim', pretrained=True, progress=True))
            if model_type == 'cheng2020_attn':
                model_list.append(zoo.cheng2020_attn(quality=i, metric='ms-ssim', pretrained=True, progress=True))
            if model_type == 'mbt2018_mean':
                model_list.append(zoo.mbt2018_mean(quality=i, metric='ms-ssim', pretrained=True, progress=True))
            if model_type == 'mbt2018':
                model_list.append(zoo.mbt2018(quality=i, metric='ms-ssim', pretrained=True, progress=True))
        model_list[i - 1].to(device)
    return model_list


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, default='/media/wang004/data/panjiajun/data/datasets/mscoco/train', 
        help="Path to the training data directory")
    aa("--val_dir", type=str, default='/media/wang004/data/panjiajun/data/datasets/mscoco/val',
        help="Path to the validation data directory")

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="checkpoints/ldm/sd/v1-inference.yaml",
       help="Path to the configuration file for the LDM model")
    aa("--ldm_ckpt", type=str, default="checkpoints/ldm/sd/sd-v1-4-full-ema.ckpt",
       help="Path to the checkpoint file for the LDM model")
    aa("--msg_decoder_path", type=str, default="checkpoints/hidden/other_dec_48b_whit.torchscript.pt",
       help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg",
       help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=0.2, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=1e9, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=10, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def get_key_list(key_num):
    file = 'key_list.txt'
    key_list = []
    key_list_tensor = torch.zeros(1, 48).to(device)
    with open(file, "r", encoding='utf-8') as f:
        for i in range(int(key_num)):
            str = f.readline()[:-1]
            key_tensor = torch.tensor([float(str[j]) for j in range(len(str))]).to(device)
            key_list_tensor = torch.cat([key_list_tensor, key_tensor.unsqueeze(0)], dim=0)
    key_list_tensor = key_list_tensor[1:]
    return key_list_tensor


def get_train_webdataset_loader(params):
    # 定义预处理流程，移除wds.decode("pil")
    preprocess = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])

    dataset = (
        wds.WebDataset("/media/wang004/data/panjiajun/data/datasets/laion2B-en-aesthetic/{00000..01412}.tar", 
        shardshuffle=True
        ,handler=wds.warn_and_continue)
        .shuffle(100)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(preprocess)
        .batched(params.batch_size)
    )
    
    # 创建DataLoader
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4)
    return dataloader


def get_test_webdataset_loader(params):
    # 定义预处理流程，移除wds.decode("pil")
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])

    dataset = (
        wds.WebDataset("/media/wang004/data/panjiajun/data/datasets/laion2B-en-aesthetic/{01413..01568}.tar", 
        shardshuffle=False
        ,handler=wds.warn_and_continue)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(preprocess)
        .batched(1)
    )
    '''
    dataset = (
        wds.WebDataset("/media/wang004/data/panjiajun/data/datasets/laion2B-en-aesthetic/{01413..01568}.tar", 
        shardshuffle=True
        ,handler=wds.warn_and_continue)
        .shuffle(100)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(preprocess)
        .batched(1)
    )
    '''
    
    # 创建DataLoader
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4)
    return dataloader


def main(params):
    # Set seeds for reproductibility
    seed = params.seed
    print('seed: ', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    tb_path = os.path.join("/media/wang004/data/panjiajun/data/tensorboard/afhq_1")
    tb_logger = tb.SummaryWriter(log_dir=tb_path)

    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    # 冒号后面的部分是类型注解,类型注解是在Python中的一种类型提示机制,用于指定变量的类型
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    param_util.copy_param(ldm_ae.decoder, ldm_ae.decoder_temp)
    ldm_ae.decoder = ldm_ae.decoder_temp
    ldm_ae.eval()
    ldm_ae.to(device)

    # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size,
                                        num_imgs=None, shuffle=True, num_workers=4, collate_fn=None)
    
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, 1, num_imgs=1000,
                                      shuffle=True, num_workers=4, collate_fn=None)
    '''
    train_loader = get_train_webdataset_loader(params)
    val_loader = get_test_webdataset_loader(params)
    '''
    
    # 通过归一化将vqgan的输出转为图像
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    # Copy the LDM decoder and finetune the copy
    # 复制解码器用于微调，将编码器和量化卷积层设置为恒等层
    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.encoder = nn.Identity()
    ldm_decoder.quant_conv = nn.Identity()
    ldm_decoder.to(device)
    for param in ldm_decoder.parameters():
        param.requires_grad = True
    for param in ldm_ae.parameters():
        param.requires_grad = False

    optim_params = utils.parse_params(params.optimizer)
    optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

    img_size = params.img_size
    key_size = 48
    
    # channels, channels_size = param_util.get_param_size(ldm_decoder.decoder)
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    wg_net = WaterMarkGenNetOneNoMask(key_size=key_size, channels_size=channels_size).to(device)
    # ke_net = KeyExtractNetPool(key_size=key_size, image_size=(3, params.img_size, params.img_size)).to(device)
    ke_net = ConvNextBase(key_size=key_size, image_size=(3, params.img_size, params.img_size)).to(device)
    
    wg_optimizer = optim.AdamW(wg_net.parameters(), lr=1e-4, weight_decay=5e-4, betas=(0.9, 0.999), amsgrad=False,
                               eps=1e-7)
    ke_optimizer = optim.AdamW(ke_net.parameters(), lr=1e-4, weight_decay=5e-4, betas=(0.9, 0.999), amsgrad=False,
                               eps=1e-7)
    '''
    model_path = 'mscoco_35'
    status = torch.load(os.path.join(params.output_dir, f"checkpoints/{model_path}/checkpoint_14.pth"))
    ldm_decoder.load_state_dict(status.get('ldm_decoder'))
    optimizer.load_state_dict(status.get('optimizer'))
    wg_net = torch.load(os.path.join(params.output_dir, f"checkpoints/{model_path}/wg_net_14.pth"))
    ke_net = torch.load(os.path.join(params.output_dir, f"checkpoints/{model_path}/ke_net_14.pth"))
    
    status = torch.load(os.path.join(params.output_dir, f"checkpoint_000.pth"))
    ldm_decoder.load_state_dict(status.get('ldm_decoder'))
    ke_net = torch.jit.load(params.msg_decoder_path).to(device)
    ke_net.eval()
    
    status = torch.load(os.path.join(params.output_dir, "checkpoint_14.pth"))
    ldm_decoder.load_state_dict(status.get('ldm_decoder'))
    optimizer.load_state_dict(status.get('optimizer'))
    wg_net = torch.load(os.path.join(params.output_dir, "wg_net_14.pth"))
    ke_net = torch.load(os.path.join(params.output_dir,"ke_net_14.pth"))
    '''
    # wg_optimizer.load_state_dict(os.path.join(params.output_dir, "wg_optimizer_7.pth"))
    # ke_optimizer.load_state_dict(os.path.join(params.output_dir, "ke_optimizer_7.pth"))
    

    keyl = []
    for i in range(20):
        keyl.append(torch.randint(0, 2, (32,), dtype=torch.float32, device=device))
    step = 0
    num = 1
    temp = 0
    step_t = -1
    key_list = get_key_list(10)
    for ii_key in range(0, params.num_keys):
        # Training loop
        print(f'>>> Training...')
        
        step, num, temp, step_t = train(train_loader, optimizer, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net,
                     wg_optimizer, ke_net, ke_optimizer, keyl, step, tb_logger, num, channels_size, step_t, img_size=img_size, 
                     key_size=key_size, temp=temp)
        save_dict = {
                'ldm_decoder': ldm_decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'params': params,
            }
        # Save checkpoint
        torch.save(save_dict, os.path.join(params.output_dir, f"c.pth"))
        torch.save(wg_net, os.path.join(params.output_dir, f"wg_net_{ii_key}.pth"))
        torch.save(ke_net, os.path.join(params.output_dir, f"ke_net_{ii_key}.pth"))
        
        '''
        if (step > 55000):
            torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint_final.pth"))
            torch.save(wg_net, os.path.join(params.output_dir, f"wg_net_final.pth"))
            torch.save(ke_net, os.path.join(params.output_dir, f"ke_net_final.pth"))
            return
        '''
        
        if ii_key % 1 == 0:
            # val_ss(val_loader, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net, ke_net, img_size=img_size, key_size=key_size, key=key_list[ii_key])
            # val_webdataset(val_loader, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net, ke_net, img_size=img_size, key_size=key_size)
            val(val_loader, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net, ke_net, img_size=img_size, key_size=key_size)
            '''
            
            key_t = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
            val_fixkey(val_loader, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net, ke_net, 
                img_size=img_size, key_size=key_size, key=key_t)
            '''
            
            # torch.save(wg_optimizer.state_dict(), os.path.join(params.output_dir, f"wg_optimizer_{ii_key}.pth"))
            # torch.save(ke_optimizer.state_dict(), os.path.join(params.output_dir, f"ke_optimizer_{ii_key}.pth"))
            
        '''
        model_type_list = ['bmshj2018_factorized', 'cheng2020_anchor', 'mbt2018']
        metric_list = ['mse']
        for mt in model_type_list:
            for me in metric_list:
                print('model_type: ', mt, '| metric: ', me)
                val_compress(val_loader, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net, ke_net, key_size, img_size,
                        channels_size, model_type=mt, metirc=me)
        '''
        print('\n')


def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL,
          vqgan_to_imnet: nn.Module, params: argparse.Namespace, wg_net,
          wg_optimizer, ke_net, ke_optimizer, keyl, step, tb_logger, num, channels_size, step_t, img_size, key_size, temp):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    wg_net.train()
    ke_net.train()
    base_lr = optimizer.param_groups[0]["lr"]
    provider = LossProvider()
    loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum').to(device)
    loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    loss_i = loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_m = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    loss_mae = lambda imgs_w, imgs: torch.mean(abs(imgs_w - imgs))
    aug = TrainDataAugmentationAll(img_size=img_size).to(device)
    n = 0
    bit_acc = 0
    temp_acc = 0
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    m_temp = torch.ones((channels_size,)).to(device)
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        if num > 1:
            utils.adjust_learning_rate(optimizer, step, params.steps, params.warmup_steps, base_lr, num, min_lr=1e-4)
        utils.adjust_learning_rate(ke_optimizer, step, params.steps, params.warmup_steps, base_lr, num, min_lr=1e-4)
        utils.adjust_learning_rate(wg_optimizer, step, params.steps, params.warmup_steps, base_lr, num, min_lr=1e-4)
        # encode images
        imgs_z = ldm_ae.encode(imgs)  # b c h w -> b z h/f w/f
        # mode：返回均值向量
        imgs_z = imgs_z.mode() 
        # 使用原始解码器解码
        w0 = gen_null_watermark(channels_size, device)
        imgs_d0 = ldm_ae.decode(imgs_z, w0)  # b z h/f w/f -> b c h w

        key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        wa, m = wg_net(key)
        # wa, b, m = wg_net(key) 
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        # 使用微调解码器解码 clamp(0,1)
        imgs_w = ldm_decoder.decode(imgs_z, w) # b z h/f w/f -> b c h w
        imgs_aug = aug(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1), num).clamp(0, 1)
        # imgs_aug = imgs_w
        extract_key = ke_net(utils_img.normalize_img(imgs_aug))

        bit_acc += losses.get_bit_acc(extract_key, key)
        # mask_rate = m.sum() / m.size(0)
        mask_rate = 0
        
        a0, a1, a2, a3 = a0_i.w, a1_i.w, 2, 0
        
        # a0, a1, a2, a3 = 1, 0, 1, 0
        '''
        if step_t == -1:
            a0, a1, a2, a3 = 1, 0, 1, 0
        else:
            a0, a1, a2, a3 = 2, 0, 0, 0
        '''
        
        # 计算损失(预测、一致性、密钥提取、掩模稀疏度)
        # wouaf使用原始图像计算
        loss_vgg = loss_i(imgs_w, imgs_d0).mean()
        loss_mse = loss_m(imgs_w, imgs_d0)
        loss_key_ext = losses.key_extract_loss(extract_key, key)
        # loss_mask = losses.mask_loss(m) * a3
        # loss_mask = losses.get_mask_coincidence_rate(m, m_temp) * a3
        # m_temp = m.detach()
        loss_mask = 0
        # 设置梯度累加次数
        accumulation_steps = 8
        loss = (loss_vgg * a0 + loss_mse * a1 + loss_key_ext * a2 + loss_mask) / accumulation_steps
        loss.backward()
        n += 1
        if bit_acc / n > 0.95 and num < 36:
            temp += 1
            if temp == 5:
                temp = 0
                num = num + 1
        if num == 36 and step_t == -1:
            step_t = step
        
        if step % 100 == 0 and n == accumulation_steps:
            print(
                "step:{:.0f}| loss: {:.4f}| loss_vgg: {:.4f}| loss_mse:{:.4f}| loss_key_ext:{:.4f}| loss_mask:{:.4f}| bit_acc:{:.4f}| mask_rate:{:.4f}"
                .format(step, loss.item(), loss_vgg.item(), loss_mse.item(), loss_key_ext.item(), loss_mask, bit_acc / n, mask_rate),
            )
            # 修改psnr计算对象
            psnr_d0_w = utils_img.psnr(imgs_w, imgs_d0).mean().item()
            psnr_orig_w = utils_img.psnr(imgs_w, imgs).mean().item()
            ssim_d0_w = ssim.ssim(imgs_w, imgs_d0).item()
            ssim_orig_w = ssim.ssim(imgs_w, imgs).item()
            print('psnr_d0_w: {:.4f}|psnr_orig_w  {:.4f}|ssim_d0_w: {:.4f}| ssim_orig_w: {:.4f}| num:{}'.format(psnr_d0_w, psnr_orig_w, ssim_d0_w, ssim_orig_w, num))
            # print('step_t: ', step_t
            # print(utils_img.psnr(imgs_d0, imgs).mean().item())

        if (ii+1) % accumulation_steps == 0 or (ii+1) == len(data_loader):
            step += 1
            if num > 1:
                optimizer.step()
                optimizer.zero_grad()
            wg_optimizer.step()
            ke_optimizer.step()
            wg_optimizer.zero_grad()
            ke_optimizer.zero_grad()

            tb_logger.add_scalar("loss", loss.item(), global_step=step)
            tb_logger.add_scalar("loss_vgg", loss_vgg.item() * 1, global_step=step)
            tb_logger.add_scalar("loss_mse", loss_mse.item() * 50, global_step=step)
            tb_logger.add_scalar("loss_key_ext", loss_key_ext.item() * 2, global_step=step)
            tb_logger.add_scalar("loss_vgg_real", loss_vgg.item() * a0_i.w, global_step=step)
            tb_logger.add_scalar("loss_mse_real", loss_mse.item() * a1_i.w, global_step=step)
            tb_logger.add_scalar("loss_key_ext_real", loss_key_ext.item() * a2, global_step=step)
            tb_logger.add_scalar("bit_acc", bit_acc / n, global_step=step)
            tb_logger.add_scalar("mask_rate", mask_rate, global_step=step)
            tb_logger.add_scalar("a0", a0_i.w, global_step=step)
            tb_logger.add_scalar("a1", a1_i.w, global_step=step)
            # tb_logger.add_scalar("loss_mask_coincidence_rate", loss_mask.item(), global_step=step)
            
            if bit_acc / n > 0.97 and temp_acc > 0.97:
                a0_i.update()
                a1_i.update()
            temp_acc = bit_acc / n
            
            '''
            if step - step_t >= 10000:
                return step, num, temp, step_t
            '''
            bit_acc = n = 0

        # save images during training
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1),
                       os.path.join(params.imgs_dir, f'train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                       os.path.join(params.imgs_dir, f'train_w.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                       os.path.join(params.imgs_dir, f'train_d0.png'), nrow=8)           
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                       os.path.join(params.imgs_dir, f'train_worig_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs)) * 5, 0, 1),
                       os.path.join(params.imgs_dir, f'train_5*worig_res.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                       os.path.join(params.imgs_dir, f'train_wd0_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0))*10, 0, 1),
                       os.path.join(params.imgs_dir, f'train_10*wd0_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0)) * 5, 0, 1),
                       os.path.join(params.imgs_dir, f'train_5*wd0_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_d0) - utils_img.unnormalize_vqgan(imgs)) * 5, 0, 1),
                       os.path.join(params.imgs_dir, f'train_d0orig_res.png'), nrow=8)
            save_image(torch.clamp(imgs_aug, 0, 1),
                       os.path.join(params.imgs_dir, f'train_w_aug.png'), nrow=8)
    return step, num, temp, step_t



def train_webdataset(data_loader, optimizer: torch.optim.Optimizer, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL,
          vqgan_to_imnet: nn.Module, params: argparse.Namespace, wg_net,
          wg_optimizer, ke_net, ke_optimizer, keyl, step, tb_logger, num, channels_size, step_t, img_size, key_size, temp):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    wg_net.train()
    ke_net.train()
    base_lr = optimizer.param_groups[0]["lr"]
    provider = LossProvider()
    loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum').to(device)
    loss_i = lambda imgs_w, imgs: loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]
    loss_m = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)
    loss_mae = lambda imgs_w, imgs: torch.mean(abs(imgs_w - imgs))
    aug = TrainDataAugmentation(img_size=img_size).to(device)
    n = 0
    bit_acc = 0
    temp_acc = 0
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    m_temp = torch.ones((channels_size,)).to(device)
    ii = 0
    for imgs in data_loader:
        imgs = imgs[0].to(device)
        ii += 1
        if num > 1:
            utils.adjust_learning_rate(optimizer, step, params.steps, params.warmup_steps, base_lr, num, min_lr=1e-4)
        utils.adjust_learning_rate(ke_optimizer, step, params.steps, params.warmup_steps, base_lr, num, min_lr=1e-4)
        utils.adjust_learning_rate(wg_optimizer, step, params.steps, params.warmup_steps, base_lr, num, min_lr=1e-4)
        # encode images
        imgs_z = ldm_ae.encode(imgs)  # b c h w -> b z h/f w/f
        # mode：返回均值向量
        imgs_z = imgs_z.mode() 
        # 使用原始解码器解码
        w0 = gen_null_watermark(channels_size, device)
        imgs_d0 = ldm_ae.decode(imgs_z, w0)  # b z h/f w/f -> b c h w

        key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        wa, m = wg_net(key)

        # wa, b, m = wg_net(key) 
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        # 使用微调解码器解码 clamp(0,1)
        imgs_w = ldm_decoder.decode(imgs_z, w) # b z h/f w/f -> b c h w
        imgs_aug = aug(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1), num).clamp(0, 1)
        # imgs_aug = utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)
        extract_key = ke_net(utils_img.normalize_img(imgs_aug))

        bit_acc += losses.get_bit_acc(extract_key, key)
        # mask_rate = m.sum() / m.size(0)
        mask_rate = 0
        
        # a0, a1, a2, a3 = a0_i.w, 0, 2, 0
        # a0, a1, a2, a3 = 1, 0, 2, 0
        a0, a1, a2, a3 = a0_i.w, a1_i.w, 2, 0
        '''
        if num == 0:
            a0, a1, a2, a3 = a0_i.w, a1_i.w, 2, 0
            # a0, a1, a2, a3 = 1, 50, 2, 0
        else:
            a0, a1, a2, a3 = 1, 50, 0, 0
        '''
        # 计算损失(预测、一致性、密钥提取、掩模稀疏度)
        # wouaf使用原始图像计算
        loss_vgg = loss_i(imgs_w, imgs_d0)
        loss_mse = loss_m(imgs_w, imgs_d0)
        loss_key_ext = losses.key_extract_loss(extract_key, key)
        # loss_mask = losses.mask_loss(m) * a3
        # loss_mask = losses.get_mask_coincidence_rate(m, m_temp) * a3
        # m_temp = m.detach()
        loss_mask = 0
        # 设置梯度累加次数
        accumulation_steps = 8
        loss = (loss_vgg * a0 + loss_mse * a1 + loss_key_ext * a2 + loss_mask) / accumulation_steps
        loss.backward()
        n += 1
        if bit_acc / n > 0.95 and num < 36:
            temp += 1
            if temp == 5:
                temp = 0
                num = num + 1
                step_t = step
        
        if step % 100 == 0 and n == accumulation_steps:
            print(
                "step:{:.0f}| loss: {:.4f}| loss_vgg: {:.4f}| loss_mse:{:.4f}| loss_key_ext:{:.4f}| loss_mask:{:.4f}| bit_acc:{:.4f}| mask_rate:{:.4f}"
                .format(step, loss.item(), loss_vgg.item(), loss_mse.item(), loss_key_ext.item(), loss_mask, bit_acc / n, mask_rate),
            )
            # 修改psnr计算对象
            psnr_d0_w = utils_img.psnr(imgs_w, imgs_d0).mean().item()
            psnr_orig_w = utils_img.psnr(imgs_w, imgs).mean().item()
            ssim_d0_w = ssim.ssim(imgs_w, imgs_d0).item()
            ssim_orig_w = ssim.ssim(imgs_w, imgs).item()
            print('psnr_d0_w: {:.4f}|psnr_orig_w  {:.4f}|ssim_d0_w: {:.4f}| ssim_orig_w: {:.4f}| num:{}'.format(psnr_d0_w, psnr_orig_w, ssim_d0_w, ssim_orig_w, num))
            # print('step_t: ', step_t
            # print(utils_img.psnr(imgs_d0, imgs).mean().item())

        if (ii+1) % accumulation_steps == 0 :
            step += 1
            if num > 1:
                optimizer.step()
                optimizer.zero_grad()
            wg_optimizer.step()
            ke_optimizer.step()
            wg_optimizer.zero_grad()
            ke_optimizer.zero_grad()
            if (step > 55000):
                return step, num, temp, step_t
            tb_logger.add_scalar("loss", loss.item(), global_step=step)
            tb_logger.add_scalar("loss_vgg", loss_vgg.item() * 1, global_step=step)
            tb_logger.add_scalar("loss_mse", loss_mse.item() * 50, global_step=step)
            tb_logger.add_scalar("loss_key_ext", loss_key_ext.item() * 2, global_step=step)
            tb_logger.add_scalar("loss_vgg_real", loss_vgg.item() * a0_i.w, global_step=step)
            tb_logger.add_scalar("loss_mse_real", loss_mse.item() * a1_i.w, global_step=step)
            tb_logger.add_scalar("loss_key_ext_real", loss_key_ext.item() * a2, global_step=step)
            tb_logger.add_scalar("bit_acc", bit_acc / n, global_step=step)
            tb_logger.add_scalar("mask_rate", mask_rate, global_step=step)
            tb_logger.add_scalar("a0", a0_i.w, global_step=step)
            tb_logger.add_scalar("a1", a1_i.w, global_step=step)
            # tb_logger.add_scalar("loss_mask_coincidence_rate", loss_mask.item(), global_step=step)
            
            if bit_acc / n > 0.97 and temp_acc > 0.97:
                a0_i.update()
                a1_i.update()
            temp_acc = bit_acc / n
            
            '''
            if step - step_t == 10000:
                return step, num, temp, step_t
            '''
            bit_acc = n = 0

        # save images during training
        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1),
                       os.path.join(params.imgs_dir, f'train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                       os.path.join(params.imgs_dir, f'train_w.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                       os.path.join(params.imgs_dir, f'train_d0.png'), nrow=8)           
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                       os.path.join(params.imgs_dir, f'train_worig_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs)) * 5, 0, 1),
                       os.path.join(params.imgs_dir, f'train_5*worig_res.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                       os.path.join(params.imgs_dir, f'train_wd0_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0))*10, 0, 1),
                       os.path.join(params.imgs_dir, f'train_10*wd0_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0)) * 5, 0, 1),
                       os.path.join(params.imgs_dir, f'train_5*wd0_res.png'), nrow=8)
            save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_d0) - utils_img.unnormalize_vqgan(imgs)) * 5, 0, 1),
                       os.path.join(params.imgs_dir, f'train_d0orig_res.png'), nrow=8)
            save_image(torch.clamp(imgs_aug, 0, 1),
                       os.path.join(params.imgs_dir, f'train_w_aug.png'), nrow=8)
    return step, num, temp, step_t


'''
'erase_0.05': lambda x: aug('Erasing', (0.05, 0.05), x),
        'erase_0.10': lambda x: aug('Erasing', (0.10, 0.10), x),
        'erase_0.15': lambda x: aug('Erasing', (0.15, 0.15), x),
        'erase_0.2': lambda x: aug('Erasing', (0.2, 0.2), x),
        'rot_10': lambda x: aug('Rotation', (10, 10), x),
        'rot_15': lambda x: aug('Rotation', (15, 15), x),
        'rot_20': lambda x: aug('Rotation', (20, 20), x),
        'rot_25': lambda x: aug('Rotation', (25, 25), x),
        'rot_30': lambda x: aug('Rotation', (30, 30), x),
        'blur_3': lambda x: aug('Blur', (3, 3), x),
        'blur_5': lambda x: aug('Blur', (5, 5), x),
        'blur_7': lambda x: aug('Blur', (7, 7), x),
        'crop_0.8': lambda x: aug('Crop', (int(img_size * 0.8), int(img_size * 0.8)), x),
        'crop_0.85': lambda x: aug('Crop', (int(img_size * 0.85), int(img_size * 0.85)), x),
        'crop_0.90': lambda x: aug('Crop', (int(img_size * 0.90), int(img_size * 0.90)), x),
        'crop_0.95': lambda x: aug('Crop', (int(img_size * 0.95), int(img_size * 0.95)), x),
        'brightness_1.3': lambda x: aug('Brightness', (1.3, 1.3), x),
        'brightness_1.2': lambda x: aug('Brightness', (1.2, 1.2), x),
        'brightness_1.1': lambda x: aug('Brightness', (1.1, 1.1), x),
        'brightness_0.9': lambda x: aug('Brightness', (0.9, 0.9), x),
        'brightness_0.8': lambda x: aug('Brightness', (0.8, 0.8), x),
        'brightness_0.7': lambda x: aug('Brightness', (0.7, 0.7), x),
        'Noise_0.05': lambda x: aug('Noise', 0.05, x),
        'Noise_0.10': lambda x: aug('Noise', 0.10, x),
        'Noise_0.15': lambda x: aug('Noise', 0.15, x),
        'Noise_0.2': lambda x: aug('Noise', 0.2, x),
        'jpeg_90': lambda x: aug('JPEG', (90, 90), x),
        'jpeg_80': lambda x: aug('JPEG', (80, 80), x),
        'jpeg_70': lambda x: aug('JPEG', (70, 70), x),
        'jpeg_60': lambda x: aug('JPEG', (60, 60), x),
        'jpeg_50': lambda x: aug('JPEG', (50, 50), x),
        'combine1': lambda x: aug('Combine', 1, x),
        'combine2': lambda x: aug('Combine', 2, x),
        'combine3': lambda x: aug('Combine', 3, x),
        'combine4': lambda x: aug('Combine', 4, x),
        'combine5': lambda x: aug('Combine', 5, x),
'''

@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, vqgan_to_imnet: nn.Module,
        params: argparse.Namespace, wg_net, ke_net, img_size, key_size):
    header = 'Eval'
    ldm_decoder.decoder.eval()
    wg_net.eval()
    ke_net.eval()
    # channels, channels_size = param_util.get_param_size(ldm_decoder.decoder)
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # att_net = torch.load(os.path.join(params.output_dir,"att_net_5.pth")).to(device)
    attacks = {
        'none': lambda x: x,
        'erase_0.05': lambda x: aug('Erasing', (0.05, 0.05), x),
        'erase_0.10': lambda x: aug('Erasing', (0.10, 0.10), x),
        'erase_0.15': lambda x: aug('Erasing', (0.15, 0.15), x),
        'erase_0.2': lambda x: aug('Erasing', (0.2, 0.2), x),
        'rot_10': lambda x: aug('Rotation', (10, 10), x),
        'rot_15': lambda x: aug('Rotation', (15, 15), x),
        'rot_20': lambda x: aug('Rotation', (20, 20), x),
        'rot_25': lambda x: aug('Rotation', (25, 25), x),
        'rot_30': lambda x: aug('Rotation', (30, 30), x),
        'blur_3': lambda x: aug('Blur', (3, 3), x),
        'blur_5': lambda x: aug('Blur', (5, 5), x),
        'blur_7': lambda x: aug('Blur', (7, 7), x),
        'crop_0.8': lambda x: aug('Crop', (int(img_size * 0.8), int(img_size * 0.8)), x),
        'crop_0.85': lambda x: aug('Crop', (int(img_size * 0.85), int(img_size * 0.85)), x),
        'crop_0.90': lambda x: aug('Crop', (int(img_size * 0.90), int(img_size * 0.90)), x),
        'crop_0.95': lambda x: aug('Crop', (int(img_size * 0.95), int(img_size * 0.95)), x),
        'brightness_1.3': lambda x: aug('Brightness', (1.3, 1.3), x),
        'brightness_1.2': lambda x: aug('Brightness', (1.2, 1.2), x),
        'brightness_1.1': lambda x: aug('Brightness', (1.1, 1.1), x),
        'brightness_0.9': lambda x: aug('Brightness', (0.9, 0.9), x),
        'brightness_0.8': lambda x: aug('Brightness', (0.8, 0.8), x),
        'brightness_0.7': lambda x: aug('Brightness', (0.7, 0.7), x),
        'Noise_0.05': lambda x: aug('Noise', 0.05, x),
        'Noise_0.10': lambda x: aug('Noise', 0.10, x),
        'Noise_0.15': lambda x: aug('Noise', 0.15, x),
        'Noise_0.2': lambda x: aug('Noise', 0.2, x),
        'jpeg_90': lambda x: aug('JPEG', (90, 90), x),
        'jpeg_80': lambda x: aug('JPEG', (80, 80), x),
        'jpeg_70': lambda x: aug('JPEG', (70, 70), x),
        'jpeg_60': lambda x: aug('JPEG', (60, 60), x),
        'jpeg_50': lambda x: aug('JPEG', (50, 50), x),
    }
    bit_acc_d0 = 0
    bit_acc = []
    for i in range(len(attacks)):
        bit_acc.append(0)
    bit_acc_list_d0 = []
    bit_acc_list = []
    for i in range(len(attacks)):
        bit_acc_list.append([])
    psnr = 0
    psnr_d0 = 0
    psnr_w_d0 = 0
    ssim_w_d0 = 0
    adv_bit_acc = 0
    all = 0
    num = []
    for i in range(len(attacks)):
        num.append(0)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    aug = TestDataAugmentation(img_size=img_size).to(device)
    avg_lpips = 0
    num_batch = 0
    '''
    pipe = ReSDPipeline.from_pretrained(pretrained_model_name_or_path=
        '/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub/stable-diffusion-2-1', 
        torch_dtype=torch.float16, revision="fp16").to(device)
    
    wmatt = VAEWMAttackerOnline(model_name='bmshj2018-factorized', quality=3, device=device)
    avg_att_acc = 0
    psnr_d0 = 0
    '''
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()
        num_batch += 1
        key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        w0 = gen_null_watermark(channels_size, device)
        imgs_d0 = ldm_ae.decode(imgs_z, w0)
        imgs_w = ldm_decoder.decode(imgs_z, w)
        '''
        229.8ms
        '''
        
        # 计算图像质量
        avg_lpips += loss_fn_vgg.forward(imgs_d0, imgs_w).mean()
        psnr += utils_img.psnr(imgs_w, imgs).mean().item()
        psnr_d0 += utils_img.psnr(imgs_d0, imgs).mean().item()
        psnr_w_d0 += utils_img.psnr(imgs_w, imgs_d0).mean().item()
        ssim_w_d0 += ssim.ssim(imgs_w, imgs_d0).item()
        all += imgs.size(0)
        
        # 从原始模型生成的图像中提取水印，计算精度
        extract_key = ke_net(utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_d0).clamp(0, 1)))
        bit_acc_d0 += losses.get_bit_acc(extract_key, key)
        bit_acc_list_d0 += losses.get_bit_acc_list(extract_key, key)
        '''
        img_att = utils_img.normalize_vqgan(wmatt.attack(utils_img.unnormalize_vqgan(imgs_w))).to(device)
        psnr += utils_img.psnr(imgs_w, img_att).mean().item()
        psnr_d0 += utils_img.psnr(imgs_d0, img_att).mean().item()
        extract_key = ke_net(img_att)
        avg_att_acc += losses.get_bit_acc(extract_key, key)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) , 0, 1),
                    os.path.join(params.imgs_dir, f'test_w.png'), nrow=8)
        save_image(torch.clamp(utils_img.unnormalize_vqgan(img_att), 0, 1),
                    os.path.join(params.imgs_dir, f'test_w_att.png'), nrow=8)
        save_image(torch.clamp(utils_img.unnormalize_vqgan(img_att) - utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                    os.path.join(params.imgs_dir, f'w_att_res.png'), nrow=8)
        
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_worig_res.png'), nrow=8)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                    os.path.join(params.imgs_dir, f'test_wd0_res.png'), nrow=8)
                    
        save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0)) * 5, 0, 1),
                    os.path.join(params.imgs_dir, f'test_5*wd0_res.png'), nrow=8)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_d0orig_res.png'), nrow=8)
                    '''
        
        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常
            # imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w))
            # imgs_aug = utils_img.normalize_vqgan(imgs_aug)
            imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
            extract_key = ke_net(utils_img.normalize_img(imgs_aug))
            acc = losses.get_bit_acc(extract_key, key)
            bit_acc[i] += acc
            # num[i] += losses.get_attribution_rate(extract_key, key)
            bit_acc_list[i] += (losses.get_bit_acc_list(extract_key, key))
            i += 1
            '''
            # 增加残差和5倍残差图像
            save_image(torch.clamp(imgs_aug, 0, 1),
                        os.path.join(params.imgs_dir, f'{name}_test.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                        os.path.join(params.imgs_dir, f'test_d0.png'), nrow=8)
            
            save_image(torch.clamp((imgs_aug - utils_img.unnormalize_vqgan(imgs_d0))*10, 0, 1),
                os.path.join(params.imgs_dir, f'10test_res{ii}.png'), nrow=8)
            '''
    '''
    print('avg_att_acc: ', avg_att_acc / num_batch)
    print('avg_psnr:', psnr / num_batch)
    print('psnr_d0:', psnr_d0 / num_batch)
    '''  
    print('avg_psnr:', psnr / num_batch)
    print('avg_psnr_d0:', psnr_d0 / num_batch)
    print('avg_psnr_w_d0:', psnr_w_d0 / num_batch)
    print('avg_ssim_w_d0:', ssim_w_d0 / num_batch)
    print('avg_lpips_w_d0:', avg_lpips / num_batch)
    print('bit_acc_d0:', bit_acc_d0 / num_batch)
    i = 0
    for name, attack in attacks.items():
        print(name, ' bit_acc:{:.4f}'.format(bit_acc[i] / num_batch))
        adv_bit_acc += bit_acc[i] / num_batch
        # print(name, ' attribution_rate:{:.4f}'.format(num[i] / all))
        auc, acc, low = losses.get_roc(bit_acc_list_d0, bit_acc_list[i])
        print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
        i += 1
    print('adv_bit_acc: ', adv_bit_acc / num_batch)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val_ss(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, vqgan_to_imnet: nn.Module,
        params: argparse.Namespace, wg_net, ke_net, img_size, key_size, key):
    header = 'Eval'
    ldm_decoder.decoder.eval()
    wg_net.eval()
    ke_net.eval()
    # channels, channels_size = param_util.get_param_size(ldm_decoder.decoder)
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # att_net = torch.load(os.path.join(params.output_dir,"att_net_5.pth")).to(device)
    attacks = {
        'none': lambda x: x,
        'erase_0.05': lambda x: aug('Erasing', (0.05, 0.05), x),
        'erase_0.10': lambda x: aug('Erasing', (0.10, 0.10), x),
        'erase_0.15': lambda x: aug('Erasing', (0.15, 0.15), x),
        'erase_0.2': lambda x: aug('Erasing', (0.2, 0.2), x),
        'rot_10': lambda x: aug('Rotation', (10, 10), x),
        'rot_15': lambda x: aug('Rotation', (15, 15), x),
        'rot_20': lambda x: aug('Rotation', (20, 20), x),
        'rot_25': lambda x: aug('Rotation', (25, 25), x),
        'rot_30': lambda x: aug('Rotation', (30, 30), x),
        'blur_3': lambda x: aug('Blur', (3, 3), x),
        'blur_5': lambda x: aug('Blur', (5, 5), x),
        'blur_7': lambda x: aug('Blur', (7, 7), x),
        'crop_0.8': lambda x: aug('Crop', (int(img_size * 0.8), int(img_size * 0.8)), x),
        'crop_0.85': lambda x: aug('Crop', (int(img_size * 0.85), int(img_size * 0.85)), x),
        'crop_0.90': lambda x: aug('Crop', (int(img_size * 0.90), int(img_size * 0.90)), x),
        'crop_0.95': lambda x: aug('Crop', (int(img_size * 0.95), int(img_size * 0.95)), x),
        'brightness_1.3': lambda x: aug('Brightness', (1.3, 1.3), x),
        'brightness_1.2': lambda x: aug('Brightness', (1.2, 1.2), x),
        'brightness_1.1': lambda x: aug('Brightness', (1.1, 1.1), x),
        'brightness_0.9': lambda x: aug('Brightness', (0.9, 0.9), x),
        'brightness_0.8': lambda x: aug('Brightness', (0.8, 0.8), x),
        'brightness_0.7': lambda x: aug('Brightness', (0.7, 0.7), x),
        'Noise_0.05': lambda x: aug('Noise', 0.05, x),
        'Noise_0.10': lambda x: aug('Noise', 0.10, x),
        'Noise_0.15': lambda x: aug('Noise', 0.15, x),
        'Noise_0.2': lambda x: aug('Noise', 0.2, x),
        'jpeg_90': lambda x: aug('JPEG', (90, 90), x),
        'jpeg_80': lambda x: aug('JPEG', (80, 80), x),
        'jpeg_70': lambda x: aug('JPEG', (70, 70), x),
        'jpeg_60': lambda x: aug('JPEG', (60, 60), x),
        'jpeg_50': lambda x: aug('JPEG', (50, 50), x),
        'combine1': lambda x: aug('Combine', 1, x),
        'combine2': lambda x: aug('Combine', 2, x),
        'combine3': lambda x: aug('Combine', 3, x),
        'combine4': lambda x: aug('Combine', 4, x),
        'combine5': lambda x: aug('Combine', 5, x),
    }
    bit_acc_d0 = 0
    bit_acc = []
    for i in range(len(attacks)):
        bit_acc.append(0)
    bit_acc_list_d0 = []
    bit_acc_list = []
    for i in range(len(attacks)):
        bit_acc_list.append([])
    psnr = 0
    psnr_d0 = 0
    psnr_w_d0 = 0
    ssim_w_d0 = 0
    adv_bit_acc = 0
    all = 0
    num = []
    for i in range(len(attacks)):
        num.append(0)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    aug = TestDataAugmentation(img_size=img_size).to(device)
    avg_lpips = 0
    num_batch = 0
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()
        num_batch += 1
        # key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        w0 = gen_null_watermark(channels_size, device)

        imgs_d0 = ldm_ae.decode(imgs_z, w0)
        w0[1] = 0
        imgs_w = ldm_decoder.decode(imgs_z, w0)
        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        # w0 = gen_null_watermark(channels_size)
        # 33.6ms
        # imgs_d0 = ldm_ae.decode(imgs_z, w0)
        imgs_w = ldm_decoder.decode(imgs_z, w)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        # 229.8ms
        print(start.elapsed_time(end))
        return
        '''
        # 计算图像质量
        avg_lpips += loss_fn_vgg.forward(imgs_d0, imgs_w).mean()
        psnr += utils_img.psnr(imgs_w, imgs).mean().item()
        psnr_d0 += utils_img.psnr(imgs_d0, imgs).mean().item()
        psnr_w_d0 += utils_img.psnr(imgs_w, imgs_d0).mean().item()
        ssim_w_d0 += ssim.ssim(imgs_w, imgs_d0).item()
        all += imgs.size(0)
        
        # 从原始模型生成的图像中提取水印，计算精度
        extract_key = ke_net(utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_d0).clamp(0, 1)))
        bit_acc_d0 += losses.get_bit_acc_stable_signature(extract_key, key)
        bit_acc_list_d0 += losses.get_bit_acc_list_stable_signature(extract_key, key)
        
        '''
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_worig_res.png'), nrow=8)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                    os.path.join(params.imgs_dir, f'test_wd0_res.png'), nrow=8)
                    
        save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0)) * 5, 0, 1),
                    os.path.join(params.imgs_dir, f'test_5*wd0_res.png'), nrow=8)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_d0orig_res.png'), nrow=8)
                    '''
        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常
            # imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w))
            # imgs_aug = utils_img.normalize_vqgan(imgs_aug)
            imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
            extract_key = ke_net(utils_img.normalize_img(imgs_aug))
            acc = losses.get_bit_acc_stable_signature(extract_key, key)
            bit_acc[i] += acc
            # num[i] += losses.get_attribution_rate(extract_key, key)
            bit_acc_list[i] += (losses.get_bit_acc_list_stable_signature(extract_key, key))
            i += 1
            # 增加残差和5倍残差图像
            '''
            save_image(torch.clamp(imgs_aug, 0, 1),
                        os.path.join(params.imgs_dir, f'{name}_test.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                        os.path.join(params.imgs_dir, f'test_d0.png'), nrow=8)
            
            save_image(torch.clamp((imgs_aug - utils_img.unnormalize_vqgan(imgs_d0))*10, 0, 1),
                os.path.join(params.imgs_dir, f'10test_res{ii}.png'), nrow=8)
            '''
            
                       

    print('avg_psnr:', psnr / num_batch)
    print('avg_psnr_d0:', psnr_d0 / num_batch)
    print('avg_psnr_w_d0:', psnr_w_d0 / num_batch)
    print('avg_ssim_w_d0:', ssim_w_d0 / num_batch)
    print('avg_lpips_w_d0:', avg_lpips / num_batch)
    print('bit_acc_d0:', bit_acc_d0 / num_batch)
    i = 0
    for name, attack in attacks.items():
        print(name, ' bit_acc:{:.4f}'.format(bit_acc[i] / num_batch))
        adv_bit_acc += bit_acc[i] / num_batch
        # print(name, ' attribution_rate:{:.4f}'.format(num[i] / all))
        auc, acc, low = losses.get_roc(bit_acc_list_d0, bit_acc_list[i])
        print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
        i += 1
    print('adv_bit_acc: ', adv_bit_acc / num_batch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def val_webdataset(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, vqgan_to_imnet: nn.Module,
        params: argparse.Namespace, wg_net, ke_net, img_size, key_size):
    header = 'Eval'
    ldm_decoder.decoder.eval()
    wg_net.eval()
    ke_net.eval()
    # channels, channels_size = param_util.get_param_size(ldm_decoder.decoder)
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # att_net = torch.load(os.path.join(params.output_dir,"att_net_5.pth")).to(device)
    attacks = {
        'none': lambda x: x
    }
    bit_acc_d0 = 0
    bit_acc = []
    for i in range(len(attacks)):
        bit_acc.append(0)
    bit_acc_list_d0 = []
    bit_acc_list = []
    for i in range(len(attacks)):
        bit_acc_list.append([])
    psnr = 0
    psnr_d0 = 0
    psnr_w_d0 = 0
    ssim_w_d0 = 0
    adv_bit_acc = 0
    all = 0
    num = []
    for i in range(len(attacks)):
        num.append(0)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    aug = TestDataAugmentation(img_size=img_size).to(device)
    num_batch = 0
    ii = 0
    lpips_cnt = 0
    for imgs in data_loader:
        if ii > 1000:
            break
        ii += 1
        imgs = imgs[0].to(device)
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()
        num_batch += 1
        key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        w0 = gen_null_watermark(channels_size, device)

        imgs_d0 = ldm_ae.decode(imgs_z, w0)
        imgs_w = ldm_decoder.decode(imgs_z, w)
        
        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        # w0 = gen_null_watermark(channels_size)
        # 33.6ms
        # imgs_d0 = ldm_ae.decode(imgs_z, w0)
        imgs_w = ldm_decoder.decode(imgs_z, w)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        # 229.8ms
        print(start.elapsed_time(end))
        return
        '''
        # 计算图像质量
        lpips_cnt += loss_fn_vgg.forward(imgs_d0, imgs_w).mean()
        psnr += utils_img.psnr(imgs_w, imgs).mean().item()
        psnr_d0 += utils_img.psnr(imgs_d0, imgs).mean().item()
        psnr_w_d0 += utils_img.psnr(imgs_w, imgs_d0).mean().item()
        ssim_w_d0 += ssim.ssim(imgs_w, imgs_d0).item()
        all += imgs.size(0)

        
        # 从原始模型生成的图像中提取水印，计算精度
        extract_key = ke_net(utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_d0).clamp(0, 1)))
        bit_acc_d0 += losses.get_bit_acc(extract_key, key)
        bit_acc_list_d0 += losses.get_bit_acc_list(extract_key, key)
        
        '''
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_worig_res.png'), nrow=8)
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                    os.path.join(params.imgs_dir, f'test_wd0_res.png'), nrow=8)
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_d0orig_res.png'), nrow=8)           
        save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0)) * 10, 0, 1),
                    os.path.join(params.imgs_dir, f'{ii}_test_10*wd0_res.png'), nrow=8)
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                    os.path.join(params.imgs_dir, f'{ii}_test_d0.png'), nrow=8)
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w), 0, 1),
                    os.path.join(params.imgs_dir, f'{ii}_test_w.png'), nrow=8)
        '''            
        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常
            # imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w))
            # imgs_aug = utils_img.normalize_vqgan(imgs_aug)
            imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
            extract_key = ke_net(utils_img.normalize_img(imgs_aug))
            acc = losses.get_bit_acc(extract_key, key)
            bit_acc[i] += acc
            # num[i] += losses.get_attribution_rate(extract_key, key)
            bit_acc_list[i] += (losses.get_bit_acc_list(extract_key, key))
            i += 1
            # 增加残差和5倍残差图像
            '''
            save_image(torch.clamp(imgs_aug, 0, 1),
                        os.path.join(params.imgs_dir, f'{name}_test.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                        os.path.join(params.imgs_dir, f'test_d0.png'), nrow=8)
            save_image(torch.clamp((imgs_aug - utils_img.unnormalize_vqgan(imgs_d0))*10, 0, 1),
                os.path.join(params.imgs_dir, f'10test_res{ii}.png'), nrow=8)
            '''               
    print('avg_psnr:', psnr / num_batch)
    print('avg_psnr_d0:', psnr_d0 / num_batch)
    print('avg_psnr_w_d0:', psnr_w_d0 / num_batch)
    print('avg_ssim_w_d0:', ssim_w_d0 / num_batch)
    print('avg_lpips_w_d0:', lpips_cnt / num_batch)
    print('bit_acc_d0:', bit_acc_d0 / num_batch)
    i = 0
    for name, attack in attacks.items():
        print(name, ' bit_acc:{:.4f}'.format(bit_acc[i] / num_batch))
        adv_bit_acc += bit_acc[i] / num_batch
        # print(name, ' attribution_rate:{:.4f}'.format(num[i] / all))
        auc, acc, low = losses.get_roc(bit_acc_list_d0, bit_acc_list[i])
        print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
        i += 1
    print('adv_bit_acc: ', adv_bit_acc / num_batch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val_ss_webdataset(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, vqgan_to_imnet: nn.Module,
        params: argparse.Namespace, wg_net, ke_net, img_size, key_size, key):
    header = 'Eval'
    ldm_decoder.decoder.eval()
    wg_net.eval()
    ke_net.eval()
    # channels, channels_size = param_util.get_param_size(ldm_decoder.decoder)
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # att_net = torch.load(os.path.join(params.output_dir,"att_net_5.pth")).to(device)
    attacks = {
        'none': lambda x: x,
        'erase_0.05': lambda x: aug('Erasing', (0.05, 0.05), x),
        'erase_0.10': lambda x: aug('Erasing', (0.10, 0.10), x),
        'erase_0.15': lambda x: aug('Erasing', (0.15, 0.15), x),
        'erase_0.2': lambda x: aug('Erasing', (0.2, 0.2), x),
        'rot_10': lambda x: aug('Rotation', (10, 10), x),
        'rot_15': lambda x: aug('Rotation', (15, 15), x),
        'rot_20': lambda x: aug('Rotation', (20, 20), x),
        'rot_25': lambda x: aug('Rotation', (25, 25), x),
        'rot_30': lambda x: aug('Rotation', (30, 30), x),
        'blur_3': lambda x: aug('Blur', (3, 3), x),
        'blur_5': lambda x: aug('Blur', (5, 5), x),
        'blur_7': lambda x: aug('Blur', (7, 7), x),
        'crop_0.8': lambda x: aug('Crop', (int(img_size * 0.8), int(img_size * 0.8)), x),
        'crop_0.85': lambda x: aug('Crop', (int(img_size * 0.85), int(img_size * 0.85)), x),
        'crop_0.90': lambda x: aug('Crop', (int(img_size * 0.90), int(img_size * 0.90)), x),
        'crop_0.95': lambda x: aug('Crop', (int(img_size * 0.95), int(img_size * 0.95)), x),
        'brightness_1.3': lambda x: aug('Brightness', (1.3, 1.3), x),
        'brightness_1.2': lambda x: aug('Brightness', (1.2, 1.2), x),
        'brightness_1.1': lambda x: aug('Brightness', (1.1, 1.1), x),
        'brightness_0.9': lambda x: aug('Brightness', (0.9, 0.9), x),
        'brightness_0.8': lambda x: aug('Brightness', (0.8, 0.8), x),
        'brightness_0.7': lambda x: aug('Brightness', (0.7, 0.7), x),
        'Noise_0.05': lambda x: aug('Noise', 0.05, x),
        'Noise_0.10': lambda x: aug('Noise', 0.10, x),
        'Noise_0.15': lambda x: aug('Noise', 0.15, x),
        'Noise_0.2': lambda x: aug('Noise', 0.2, x),
        'jpeg_90': lambda x: aug('JPEG', (90, 90), x),
        'jpeg_80': lambda x: aug('JPEG', (80, 80), x),
        'jpeg_70': lambda x: aug('JPEG', (70, 70), x),
        'jpeg_60': lambda x: aug('JPEG', (60, 60), x),
        'jpeg_50': lambda x: aug('JPEG', (50, 50), x),
        'combine1': lambda x: aug('Combine', 1, x),
        'combine2': lambda x: aug('Combine', 2, x),
        'combine3': lambda x: aug('Combine', 3, x),
        'combine4': lambda x: aug('Combine', 4, x),
        'combine5': lambda x: aug('Combine', 5, x),
    }
    bit_acc_d0 = 0
    bit_acc = []
    for i in range(len(attacks)):
        bit_acc.append(0)
    bit_acc_list_d0 = []
    bit_acc_list = []
    for i in range(len(attacks)):
        bit_acc_list.append([])
    psnr = 0
    psnr_d0 = 0
    psnr_w_d0 = 0
    ssim_w_d0 = 0
    adv_bit_acc = 0
    all = 0
    num = []
    for i in range(len(attacks)):
        num.append(0)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    aug = TestDataAugmentation(img_size=img_size).to(device)
    avg_lpips = 0
    avg_lpips_u = 0
    avg_lpips_i = 0
    num_batch = 0
    ii = 0
    for imgs in data_loader:
        if ii > 1000:
            break
        ii += 1
        imgs = imgs[0].to(device)
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()
        num_batch += 1
        # key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        w0 = gen_null_watermark(channels_size, device)

        imgs_d0 = ldm_ae.decode(imgs_z, w0)
        w0[1] = 0
        imgs_w = ldm_decoder.decode(imgs_z, w0)
        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        # w0 = gen_null_watermark(channels_size)
        # 33.6ms
        # imgs_d0 = ldm_ae.decode(imgs_z, w0)
        imgs_w = ldm_decoder.decode(imgs_z, w)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        # 229.8ms
        print(start.elapsed_time(end))
        return
        '''
        # 计算图像质量
        avg_lpips += loss_fn_vgg.forward(imgs_d0, imgs_w).mean()
        avg_lpips_u += loss_fn_vgg.forward(utils_img.unnormalize_vqgan(imgs_d0), utils_img.unnormalize_vqgan(imgs_w)).mean()
        avg_lpips_i += loss_fn_vgg.forward(utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_d0)), 
            utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_w))).mean()
        psnr += utils_img.psnr(imgs_w, imgs).mean().item()
        psnr_d0 += utils_img.psnr(imgs_d0, imgs).mean().item()
        psnr_w_d0 += utils_img.psnr(imgs_w, imgs_d0).mean().item()
        ssim_w_d0 += ssim.ssim(imgs_w, imgs_d0).item()
        all += imgs.size(0)
        '''
        # 从原始模型生成的图像中提取水印，计算精度
        extract_key = ke_net(utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_d0).clamp(0, 1)))
        bit_acc_d0 += losses.get_bit_acc_stable_signature(extract_key, key)
        bit_acc_list_d0 += losses.get_bit_acc_list_stable_signature(extract_key, key)
        '''
        '''
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_worig_res.png'), nrow=8)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                    os.path.join(params.imgs_dir, f'test_wd0_res.png'), nrow=8)
                    
        save_image(torch.clamp((utils_img.unnormalize_vqgan(imgs_w) - utils_img.unnormalize_vqgan(imgs_d0)) * 5, 0, 1),
                    os.path.join(params.imgs_dir, f'test_5*wd0_res.png'), nrow=8)
        
        save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0) - utils_img.unnormalize_vqgan(imgs), 0, 1),
                    os.path.join(params.imgs_dir, f'test_d0orig_res.png'), nrow=8)
                    '''
        '''
        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常
            # imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w))
            # imgs_aug = utils_img.normalize_vqgan(imgs_aug)
            imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
            extract_key = ke_net(utils_img.normalize_img(imgs_aug))
            acc = losses.get_bit_acc_stable_signature(extract_key, key)
            # print(acc)
            bit_acc[i] += acc
            # num[i] += losses.get_attribution_rate(extract_key, key)
            bit_acc_list[i] += (losses.get_bit_acc_list_stable_signature(extract_key, key))
            i += 1
            # 增加残差和5倍残差图像
            
            save_image(torch.clamp(imgs_aug, 0, 1),
                        os.path.join(params.imgs_dir, f'{name}_test.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                        os.path.join(params.imgs_dir, f'test_d0.png'), nrow=8)
            
            save_image(torch.clamp((imgs_aug - utils_img.unnormalize_vqgan(imgs_d0))*10, 0, 1),
                os.path.join(params.imgs_dir, f'10test_res{ii}.png'), nrow=8)
            '''

    print('avg_psnr:', psnr / num_batch)
    print('avg_psnr_d0:', psnr_d0 / num_batch)
    print('avg_psnr_w_d0:', psnr_w_d0 / num_batch)
    print('avg_ssim_w_d0:', ssim_w_d0 / num_batch)
    print('avg_lpips_w_d0:', avg_lpips / num_batch)
    print('avg_lpips_u_d0:', avg_lpips_u / num_batch)
    print('avg_lpips_i_d0:', avg_lpips_i / num_batch)
    print('bit_acc_d0:', bit_acc_d0 / num_batch)
    i = 0
    '''
    for name, attack in attacks.items():
        print(name, ' bit_acc:{:.4f}'.format(bit_acc[i] / num_batch))
        adv_bit_acc += bit_acc[i] / num_batch
        # print(name, ' attribution_rate:{:.4f}'.format(num[i] / all))
        auc, acc, low = losses.get_roc(bit_acc_list_d0, bit_acc_list[i])
        print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
        i += 1
    print('adv_bit_acc: ', adv_bit_acc / num_batch)
    '''
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def val_fixkey(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, vqgan_to_imnet: nn.Module,
        params: argparse.Namespace, wg_net, ke_net, img_size, key_size, key):
    header = 'Eval'
    ldm_decoder.decoder.eval()
    wg_net.eval()
    ke_net.eval()
    # channels, channels_size = param_util.get_param_size(ldm_decoder.decoder)
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    metric_logger = utils.MetricLogger(delimiter="  ")
    attacks = {
        'none': lambda x: x,
    }
    bit_acc_d0 = 0
    bit_acc = []
    for i in range(len(attacks)):
        bit_acc.append(0)
    bit_acc_list_d0 = []
    bit_acc_list = []
    for i in range(len(attacks)):
        bit_acc_list.append([])
    psnr = 0
    psnr_d0 = 0
    psnr_w_d0 = 0
    ssim_w_d0 = 0
    adv_bit_acc = 0
    all = 0
    num = []
    for i in range(len(attacks)):
        num.append(0)
    aug = TestDataAugmentation(img_size=img_size).to(device)
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()

        # key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        wa, m = wg_net(key)
        w.append(wa)
        # w.append(b)
        # w.append(len(channels))
        w.append(0)
        w.append(0)
        w0 = gen_null_watermark(channels_size, device)

        imgs_d0 = ldm_ae.decode(imgs_z, w0)
        imgs_w = ldm_decoder.decode(imgs_z, w)

        # 计算图像质量
        psnr += utils_img.psnr(imgs_w, imgs).mean().item()
        psnr_d0 += utils_img.psnr(imgs_d0, imgs).mean().item()
        psnr_w_d0 += utils_img.psnr(imgs_w, imgs_d0).mean().item()
        ssim_w_d0 += ssim.ssim(imgs_w, imgs_d0).item()
        all += imgs.size(0)

        # 从原始模型生成的图像中提取水印，计算精度
        '''
        extract_key = ke_net(utils_img.normalize_img(utils_img.unnormalize_vqgan(imgs_d0).clamp(0, 1)))
        bit_acc_d0 += losses.get_bit_acc(extract_key, key)
        bit_acc_list_d0 += losses.get_bit_acc_list(extract_key, key)
        '''
        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常
            # imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w))
            # imgs_aug = utils_img.normalize_vqgan(imgs_aug)
            imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
            extract_key = ke_net(utils_img.normalize_img(imgs_aug))
            acc = losses.get_bit_acc(extract_key, key)
            bit_acc[i] += acc
            # num[i] += losses.get_attribution_rate(extract_key, key)
            bit_acc_list[i] += (losses.get_bit_acc_list(extract_key, key))
            i += 1

    print('avg_psnr:', psnr / len(data_loader))
    print('avg_psnr_d0:', psnr_d0 / len(data_loader))
    print('avg_psnr_w_d0:', psnr_w_d0 / len(data_loader))
    print('avg_ssim_w_d0:', ssim_w_d0 / len(data_loader))
    print('bit_acc_d0:', bit_acc_d0 / len(data_loader))
    i = 0
    for name, attack in attacks.items():
        print(name, ' bit_acc:{:.4f}'.format(bit_acc[i] / len(data_loader)))
        adv_bit_acc += bit_acc[i] / len(data_loader)
        # print(name, ' attribution_rate:{:.4f}'.format(num[i] / all))
        # auc, acc, low = losses.get_roc(bit_acc_list_d0, bit_acc_list[i])
        # print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
        i += 1
    print('adv_bit_acc: ', adv_bit_acc / len(attacks))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val_collusion(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, vqgan_to_imnet: nn.Module,
        params: argparse.Namespace, wg_net, ke_net, img_size, key_size):
    '''
    ext00: 0.0004
    ext01: 0.5190
    ext10: 0.5186
    ext11: 0.9995
    '''
    header = 'Eval'
    ldm_decoder.decoder.eval()
    wg_net.eval()
    ke_net.eval()
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    metric_logger = utils.MetricLogger(delimiter="  ")
    attacks = {
        'none': lambda x: x,
    }
    bit_acc_d0 = 0
    bit_acc = []
    for i in range(len(attacks)):
        bit_acc.append(0)
    bit_acc_list_d0 = []
    bit_acc_list = []
    for i in range(len(attacks)):
        bit_acc_list.append([])
    adv_bit_acc = 0
    all = 0
    num = []
    ext00 = 0
    ext01 = 0
    ext10 = 0
    ext11 = 0
    num_ext = [0,0,0,0]
    for i in range(len(attacks)):
        num.append(0)
    aug = TestDataAugmentation(img_size=img_size).to(device)
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()

        key1 = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        key2 = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w1 = []
        w2 = []
        wa, m = wg_net(key1)
        w1.append(wa)
        w1.append(0)
        w1.append(0)
        wa, m = wg_net(key2)
        w2.append(wa)
        w2.append(0)
        w2.append(0)

        imgs_w1 = ldm_decoder.decode(imgs_z, w1)
        imgs_w2 = ldm_decoder.decode(imgs_z, w2)

        imgs_w = (imgs_w1 + imgs_w2) / 2

        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常
            # imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w))
            # imgs_aug = utils_img.normalize_vqgan(imgs_aug)
            imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
            extract_key = torch.round(ke_net(utils_img.normalize_img(imgs_aug))).reshape(-1)
            for num in range(0, key_size):
                if key1[num] == 0 and key2[num] == 0: 
                    ext00 += extract_key[num]
                    num_ext[0] += 1
                if key1[num] == 0 and key2[num] == 1: 
                    ext01 += extract_key[num]
                    num_ext[1] += 1
                if key1[num] == 1 and key2[num] == 0: 
                    ext10 += extract_key[num]
                    num_ext[2] += 1
                if key1[num] == 1 and key2[num] == 1: 
                    ext11 += extract_key[num]
                    num_ext[3] += 1
            i += 1

    print('ext00: ', ext00 / num_ext[0])
    print('ext01: ', ext01 / num_ext[1])
    print('ext10: ', ext10 / num_ext[2])
    print('ext11: ', ext11 / num_ext[3])
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val_compress(data_loader, ldm_ae, ldm_decoder, vqgan_to_imnet, params, wg_net, ke_net, key_size, img_size,
                 channels_size, model_type='mbt2018', metirc='ms-ssim'):
    header = 'Eval'
    ldm_decoder.decoder.eval()
    ke_net.eval()
    wg_net.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    model_list = get_compress_model_list(model_type=model_type, metirc=metirc)
    bit_acc_d0 = []
    bit_acc = []
    bit_acc_list_d0 = []
    bit_acc_list = []
    psnr_d0_w = []
    psnr_orig_w = []
    ssim_d0_w = []
    ssim_orig_w = []
    for j in range(len(model_list)):
        bit_acc_d0.append(0)
        bit_acc_list_d0.append([])
        bit_acc.append([])
        bit_acc_list.append([])
        for i in range(20):
            bit_acc[j].append(0)
            bit_acc_list[j].append([])
        psnr_d0_w.append(0)
        psnr_orig_w.append(0)
        ssim_d0_w.append(0)
        ssim_orig_w.append(0)
    all = 0
    num = []
    for i in range(20):
        num.append(0)
    aug = TestDataAugmentation(img_size=img_size,).to(device)
    attacks = {
        'none': lambda x: x,
    }
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        imgs_z = ldm_ae.encode(imgs).mode()
        key = torch.randint(0, 2, (key_size,), dtype=torch.float32, device=device)
        w = []
        wa, m = wg_net(key)
        w.append(wa)
        w.append(0)
        w.append(0)
        w0 = gen_null_watermark(channels_size, device=device)
        imgs_d0 = ldm_ae.decode(imgs_z, w0)
        # w0[1] = 0
        imgs_w = ldm_decoder.decode(imgs_z, w)

        all += imgs.size(0)

        # 从原始模型生成的图像中提取水印，计算精度
        for j in range(len(model_list)):
            extract_key = ke_net(model_list[j](utils_img.unnormalize_vqgan(imgs_d0))['x_hat'])
            bit_acc_d0[j] += losses.get_bit_acc(extract_key, key)
            # bit_acc_d0[j] += losses.get_bit_acc_stable_signature(extract_key, key)
            bit_acc_list_d0[j] += losses.get_bit_acc_list(extract_key, key)
            # bit_acc_list_d0[j] += losses.get_bit_acc_list_stable_signature(extract_key, key)
            # 计算图像质量
            img_w_compress = utils_img.normalize_vqgan(model_list[j](utils_img.unnormalize_vqgan(imgs_w))['x_hat'])
            psnr_d0_w[j] += utils_img.psnr(img_w_compress, imgs_d0).mean().item()
            psnr_orig_w[j] += utils_img.psnr(img_w_compress, imgs).mean().item()
            ssim_d0_w[j] += ssim.ssim(img_w_compress, imgs_d0).item()
            ssim_orig_w[j] += ssim.ssim(img_w_compress, imgs).item()

        i = 0
        for name, attack in attacks.items():
            # 转换图像格式，否则jpeg压缩会异常 
            for j in range(len(model_list)):
                imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
                imgs_aug = model_list[j](imgs_aug)['x_hat']
                extract_key = ke_net(utils_img.normalize_img(imgs_aug))
                bit_acc[j][i] += losses.get_bit_acc(extract_key, key)
                # bit_acc[j][i] += losses.get_bit_acc_stable_signature(extract_key, key)
                bit_acc_list[j][i] += (losses.get_bit_acc_list(extract_key, key))
                # bit_acc_list_d0[j] += losses.get_bit_acc_list_stable_signature(extract_key, key)
                if ii == 0:
                    # 增加残差和5倍残差图像
                    save_image(torch.clamp(imgs_aug, 0, 1),
                               os.path.join(params.imgs_dir, f'{name}_test_{j + 1}.png'), nrow=8)
                    save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                               os.path.join(params.imgs_dir, f'test_d0.png'), nrow=8)
                    if name == 'none':
                        save_image(
                            torch.clamp((imgs_aug) - utils_img.unnormalize_vqgan(imgs_d0), 0, 1),
                            os.path.join(params.imgs_dir, f'test_wd0_res.png'), nrow=8)
                        save_image(
                            torch.clamp(((imgs_aug) - utils_img.unnormalize_vqgan(imgs_d0)) * 5, 0, 1),
                            os.path.join(params.imgs_dir, f'test_5wd0_res.png'), nrow=8)
            i += 1
    for j in range(len(model_list)):
        print('compress quality: ', j + 1)
        print('avg_psnr_d0_w:', psnr_d0_w[j] / len(data_loader))
        # print('avg_psnr_orig_w:', psnr_orig_w[j] / len(data_loader))
        # print('avg_ssim_d0_w:', ssim_d0_w[j] / len(data_loader))
        # print('avg_ssim_orig_w:', ssim_orig_w[j] / len(data_loader))
        # print('bit_acc_d0:', bit_acc_d0[j] / len(data_loader))
        i = 0
        for name, attack in attacks.items():
            print(name, ' bit_acc:{:.4f}'.format(bit_acc[j][i] / len(data_loader)))
            # auc, acc, low = losses.get_roc(bit_acc_list_d0[j], bit_acc_list[j][i])
            # print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
            i += 1
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    
    # run experiment
    main(params)
