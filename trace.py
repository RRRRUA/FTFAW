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

import utils
import utils_img
import utils_model
import torch.utils.tensorboard as tb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from src.ldm.models.diffusion.ddpm import LatentDiffusion
from watermark.model.watermark_net import WaterMarkGenNet, KeyExtractNet, gen_null_watermark, KeyExtractNet_patchgan, \
    WaterMarkGenNet_NoMask, KeyExtractNet_Hidden, KeyExtractNetPool
from watermark.util import param_util
from watermark.losses import losses
from sklearn import preprocessing

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 用户识别性能检测过程：
# 1、生成：从不同容量（1e4~1e6）的用户集中随机抽取1000个密钥（用户），对于每个密钥生成5个图像，得到5000个水印图像
# 2、检测：从图像中提取的密钥，在总用户集中找到匹配度最高的用户，匹配正确则分值+1，用分值除以总图像数，得到归因正确率
# 方案来源：A Watermark-Conditioned Diffusion Model for IP Protection

def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, default='/media/wang004/data/panjiajun/data/datasets/mscoco/train', 
        help="Path to the training data directory")
    aa("--val_dir", type=str, default='/media/wang004/data/panjiajun/data/datasets/mscoco/val',
        help="Path to the validation data directory")
    aa("--ldm_config", type=str, default="checkpoints/ldm/sd/v1-inference.yaml",
       help="Path to the configuration file for the LDM model")
    aa("--ldm_ckpt", type=str, default="checkpoints/ldm/sd/sd-v1-4-full-ema.ckpt",
       help="Path to the checkpoint file for the LDM model")
    aa("--msg_decoder_path", type=str, default="models/hidden/dec_48b_whit.torchscript.pt",
       help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=1, help="Batch size for training")
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
    aa("--num_keys", type=int, default=100, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def main(params):
    # Set seeds for reproductibility
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)

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
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size, num_imgs=5000,
                                      shuffle=True, num_workers=4, collate_fn=None)

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

    key_size = 48
    # 设置用户池的容量1e4、1e5、1e6
    # num_list = [10000, 100000, 1000000]
    num_list = [1000000]
    # 生成总用户密钥集合，数量为1e6
    gen_key_list(key_size=key_size, key_num=1000000)
    model_path = ['checkpoints/mscoco_19/']
    for i in range(len(model_path)):
        print(model_path[i])
        status = torch.load(os.path.join(params.output_dir, model_path[i] + "checkpoint_14.pth"))
        ldm_decoder.load_state_dict(status.get('ldm_decoder'))
        optimizer.load_state_dict(status.get('optimizer'))
        wg_net = torch.load(os.path.join(params.output_dir, model_path[i] + "wg_net_14.pth")).to(device)
        ke_net = torch.load(os.path.join(params.output_dir, model_path[i] + "ke_net_14.pth")).to(device)

        tracing_rate = 0
        # 遍历不同容量的用户集
        for ii in range(len(num_list)):
            key_list, rand_list = get_key_list(num_list[ii], user_num=1000)
            print('11111111111')
            val(val_loader, ldm_ae, ldm_decoder, params, wg_net, ke_net, key_size=key_size, img_size=params.img_size,
                key_list=key_list, rand_list=rand_list)
            print('avg tracing rate: {:.4f} in {:.0f} user set'.format(tracing_rate, num_list[ii]))

        # tracing_rate /= len(num_list)
        # 输出平均检索成功率
        # print('avg tracing rate: {:.4f}'.format(tracing_rate))


# 根据key尺寸和数量生成用户密钥集
def gen_key_list(key_size, key_num):
    file = 'key_list.txt'
    with open(file, "w", encoding='utf-8') as f:
        for i in range(int(key_num)):
            random_array = np.random.randint(0, 2, size=(key_size,))
            str = np.array2string(random_array, separator='', max_line_width=key_size + 2)
            str = str[1:-1] + "\n"
            f.writelines(str)
    return


# 根据给定的集合大小，从现有的用户密钥集中读取key_num大小的密钥子集,并返回user_num个随机抽样的密钥索引（默认1000）
def get_key_list(key_num, user_num=1000):
    file = 'key_list.txt'
    key_list = []
    key_list_tensor = torch.zeros(1, 48).to(device)
    with open(file, "r", encoding='utf-8') as f:
        for i in range(int(key_num)):
            str = f.readline()[:-1]
            key_tensor = torch.tensor([float(str[j]) for j in range(len(str))]).to(device)
            # key_list.append(key_tensor)
            key_list_tensor = torch.cat([key_list_tensor, key_tensor.unsqueeze(0)], dim=0)
    # 不放回地从用户集中随机抽样1000个用户密钥位置
    rand_list = list(np.random.choice(int(key_num), size=user_num, replace=False))
    key_list_tensor = key_list_tensor[1:]
    return key_list_tensor, rand_list


@torch.no_grad()
def val(data_loader, ldm_ae, ldm_decoder, params, wg_net, ke_net, key_size, img_size, key_list, rand_list):
    with torch.no_grad():
        header = 'Eval'
        ldm_decoder.decoder.eval()
        wg_net.eval()
        ke_net.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        # aug = TestDataAugmentationCuda(img_size=img_size, device=device).to(device)
        attacks = {
            'none': lambda x: x,
        }
        ext_key_list = []
        correct = []
        for i in range(len(attacks)):
            ext_key_list.append([])
            correct.append(0)
        i = -1
        channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
        # 对于每个用户密钥，生成5张水印图像并从中提取密钥，得到提取密钥集ext_key_list
        for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
            imgs = imgs.to(device)
            imgs_z = ldm_ae.encode(imgs)
            imgs_z = imgs_z.mode()
            '''
            if ii % 5 == 0:
                i += 1
                if i >= len(rand_list):
                    break
            key = key_list[rand_list[i]].to(device)
            w = []
            wa, m = wg_net(key)
            w.append(wa)
            w.append(0)
            w.append(0)
            imgs_w = ldm_decoder.decode(imgs_z, w)
            '''
            # w0 = gen_null_watermark(channels_size, device)
            imgs_w = imgs
            e = 0
            for name, attack in attacks.items():
                imgs_aug = attack(utils_img.unnormalize_vqgan(imgs_w).clamp(0, 1)).clamp(0, 1)
                ext_key = ke_net(utils_img.normalize_img(imgs_aug))
                ext_key_list[e].append(ext_key)
                e += 1
        print('tracing')
        # 遍历提取密钥集，遍历用户池中检索匹配度最高的密钥，并判断是否是真实密钥
        '''
        for i in range(len(ext_key_list[0])):
            for e in range(len(attacks)):
                if torch.argmin(torch.abs(ext_key_list[e][i] - key_list).sum(1)) == rand_list[int(i / 5)]:
                    correct[e] += 1
        e = 0
        for name, attack in attacks.items():
            print('attack: {}| set_size: {}| tracing rate: {:.4f}'.format(name, len(key_list),
                                                                          correct[e] / len(ext_key_list[0])))
            e += 1
        '''
        avg_ext_key = torch.zeros(1, 48).to(device)
        for i in range(len(ext_key_list[0])):
            avg_ext_key += ext_key_list[0][i]
        avg_ext_key /= len(data_loader)
        print(torch.mean(avg_ext_key))
        return


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)
