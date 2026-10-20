import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from torchvision import models
import os
os.environ['TORCH_HOME']='/media/wang004/data/panjiajun/checkpoints'

channel = [128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
           256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
           256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
           256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 128]


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            # 设置左右上下的填充次数
            pad = (0, 1, 0, 1)
            # 填充后卷积，令输出的尺寸为原始的一半
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        # temb = None
        # h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# 修改水印格式
def size_modify(channels, x):
    target = []
    xx = []
    k = 0
    for i in range(len(channels)):
        for j in range(channels[i]):
            xx.append(x[k])
            k += 1
        target.append(xx)
        xx = []
    return target


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# 获得值全为1的空水印，调制后不影响模型权重
def gen_null_watermark(channels_size, device):
    w = []
    w.append(torch.ones(channels_size).to(device))
    # w.append(torch.zeros(channels_size).to('cuda'))
    w.append(0)
    return w


def gen_null_watermark_bias(channels_size, device):
    w = []
    w.append(torch.ones(channels_size).to(device))
    w.append(torch.zeros(channels_size).to(device))
    w.append(0)
    w.append(0)
    return w


'''
def gen_bias_watermark(channels_size, pos, wa, b):
    w = []
    if pos == 'all':
        w.append(wa)
        w.append(b)
    else:
        m =
    w.append(0)
    return  w
'''


class EqualizedWeight(nn.Module):

    def __init__(self, shape):
        super().__init__()

        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class WaterMarkGenNet(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        self.channels_size = channels_size
        self.base_mask = torch.ones(channels_size).cuda()
        layers = []
        layers.append(EqualizedLinear(key_size, 512))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        self.conv1 = nn.Sequential(*layers)
        fc1 = []
        fc2 = []
        for i in range(2):
            fc1.append(EqualizedLinear(512, 512))
            fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
            fc2.append(EqualizedLinear(512, 512))
            fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))

        fc1.append(EqualizedLinear(512, channels_size))
        fc2.append(EqualizedLinear(512, channels_size))
        self.fc1 = nn.Sequential(*fc1)
        self.fc2 = nn.Sequential(*fc2)

    def forward(self, x):
        x = self.conv1(x)
        w = self.fc1(x)
        m = self.fc2(x) + 1
        m = ste_round(F.sigmoid(m))
        w = w * m + (1 - m)
        return w, m


class WaterMarkGenNetOneNoMask(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        self.channels_size = channels_size
        layers = []
        layers.append(EqualizedLinear(key_size, 512))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        self.conv1 = nn.Sequential(*layers)
        fc1 = []
        for i in range(2):
            fc1.append(EqualizedLinear(512, 512))
            fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))

        # 令w初值为0，加1后初值为1
        output1 = EqualizedLinear(512, channels_size)
        fc1.append(output1)
        self.fc1 = nn.Sequential(*fc1)

    def forward(self, x):
        x = self.conv1(x)
        w = self.fc1(x) + 1
        m = []
        return w, m


# 生成水印系数和权重，不生成掩模，复现论文Robust Retraining-free GAN Fingerprinting via Personalized Normalization
class WaterMarkGenNetBiasNoMask(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        self.channels_size = channels_size
        fc1 = []
        fc2 = []
        fc1.append(EqualizedLinear(key_size, 512))
        fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc2.append(EqualizedLinear(key_size, 512))
        fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        for i in range(2):
            fc1.append(EqualizedLinear(512, 512))
            fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
            fc2.append(EqualizedLinear(512, 512))
            fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))

        fc1.append(EqualizedLinear(512, channels_size))
        fc2.append(EqualizedLinear(512, channels_size))
        self.fc1 = nn.Sequential(*fc1)
        self.fc2 = nn.Sequential(*fc2)

    def forward(self, x):
        w = self.fc1(x) + 1 
        b = self.fc2(x)
        return w, b, []


class WaterMarkGenNet_Bias(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        self.channels_size = channels_size
        fc1 = []
        fc2 = []
        fc3 = []
        fc1.append(EqualizedLinear(key_size, 512))
        fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc2.append(EqualizedLinear(key_size, 512))
        fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc3.append(EqualizedLinear(key_size, 512))
        fc3.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        for i in range(2):
            fc1.append(EqualizedLinear(512, 512))
            fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
            fc2.append(EqualizedLinear(512, 512))
            fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
            fc3.append(EqualizedLinear(512, 512))
            fc3.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))

        fc1.append(EqualizedLinear(512, channels_size))
        fc2.append(EqualizedLinear(512, channels_size))
        fc3.append(EqualizedLinear(512, channels_size))
        self.fc1 = nn.Sequential(*fc1)
        self.fc2 = nn.Sequential(*fc2)
        self.fc3 = nn.Sequential(*fc2)

    def forward(self, x):
        w = self.fc1(x)
        b = self.fc2(x)
        m = self.fc3(x)
        m = ste_round(F.sigmoid(m))
        w = w * m + (1 - m)
        b = b * m
        return w, b, m


class WaterMarkGenNet_NoMask(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        self.channels_size = channels_size
        layers = []
        layers.append(EqualizedLinear(key_size, 512))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        self.conv1 = nn.Sequential(*layers)
        fc1 = []
        for i in range(2):
            fc1.append(EqualizedLinear(512, 512))
            fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))

        fc1.append(EqualizedLinear(512, channels_size))
        self.fc1 = nn.Sequential(*fc1)

    def forward(self, x):
        x = self.conv1(x)
        w = self.fc1(x)
        m = []
        return w, m


def ste_round(x):
    return torch.round(x) - x.detach() + x


# 从生成图像中提取二进制密钥信息
class KeyExtractNet(nn.Module):
    def __init__(self, key_size, image_size):
        super().__init__()
        in_channels = image_size[0]
        layers = []
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        num = 1
        # 32 64 128 256
        for i in range(1, 5):
            layers.append(ResnetBlock(in_channels=32 * num, out_channels=32 * 2 * num, temb_channels=512, dropout=0.1))
            layers.append(Downsample(in_channels=32 * 2 * num, with_conv=True))
            num *= 2
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(int((image_size[1] / 16 * image_size[2] / 16) * 512), 512)
        self.fc2 = nn.Linear(512, key_size)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(self.fc1(x))
        x = F.sigmoid(x)
        return x


class KeyExtractNetPool(nn.Module):
    def __init__(self, key_size, image_size):
        super().__init__()
        in_channels = image_size[0]
        layers = []
        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        num = 1
        # 32 64 128 256
        for i in range(1, 5):
            layers.append(ResnetBlock(in_channels=32 * num, out_channels=32 * 2 * num, temb_channels=512, dropout=0.1))
            layers.append(Downsample(in_channels=32 * 2 * num, with_conv=True))
            num *= 2
        layers.append(ResnetBlock(in_channels=32 * num, out_channels=key_size, temb_channels=512, dropout=0.1))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(key_size, key_size)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x
    

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNextBase(nn.Module):
    def __init__(self, key_size, image_size):
        super().__init__()
        self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights)
        layers = []
        layers.append(LayerNorm2d(1024, eps=1e-6, elementwise_affine=True))
        layers.append(nn.Flatten(1))
        layers.append(nn.Linear(in_features=1024, out_features=key_size, bias=True))
        self.model.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = F.sigmoid(x)
        return x


class KeyExtractNet_patchgan(nn.Module):
    # initializers
    def __init__(self, d=64, image_size=(3, 32, 23), key_size=128):
        super(KeyExtractNet_patchgan, self).__init__()
        img_len = image_size[2]
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)  # 16 64
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)  # 8 32
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)  # 4 16
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)  # 3 15
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 1, 1)  # 2 14
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.fc = nn.Linear(int(d * 8 * (img_len / 8 - 2) ** 2), key_size)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class KeyExtractNet_Hidden(nn.Module):
    def __init__(self, num_blocks, num_bits, channels):
        super(KeyExtractNet_Hidden, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):
        x = self.layers(img_w)  # b d 1 1
        x = x.view(x.size(0), -1)  # b d
        x = self.linear(x)  # b d
        x = F.sigmoid(x)
        return x


def mask_gen_random(channels_size, groups):
    mask = torch.zeros((groups, channels_size))
    num = math.floor(channels_size / groups) * groups
    for i in range(num):
        t = torch.randint(0, 10, size=(1,))
        mask[int(t)][i] = 1
    return mask


class WaterMarkGenNetWithMask(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        fc1 = []
        fc2 = []
        fc3 = []
        fc1.append(EqualizedLinear(key_size, 512))
        fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc2.append(EqualizedLinear(channels_size, 512))
        fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        for i in range(2):
            fc1.append(EqualizedLinear(512, 512))
            fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
            fc2.append(EqualizedLinear(512, 512))
            fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc3.append(EqualizedLinear(1024, 1024))
        fc3.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc3.append(EqualizedLinear(1024, channels_size))
        self.fc1 = nn.Sequential(*fc1)
        self.fc2 = nn.Sequential(*fc2)
        self.fc3 = nn.Sequential(*fc3)

    def forward(self, x, m):
        w = self.fc1(x)
        m_embed = self.fc2(m)
        w = self.fc3(torch.cat([w, m_embed], dim=0)) + 1
        w = w * m
        return w, m

class ConvNextBaseWithMask(nn.Module):
    def __init__(self, key_size, channels_size):
        super().__init__()
        self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights)
        layers = []
        layers.append(LayerNorm2d(1024, eps=1e-6, elementwise_affine=True))
        layers.append(nn.Flatten(1))
        layers.append(nn.Linear(in_features=1024, out_features=512, bias=True))
        self.model.classifier = nn.Sequential(*layers)
        fc1 = []
        fc1.append(EqualizedLinear(channels_size, 512))
        fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc1.append(EqualizedLinear(512, 512))
        fc1.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc2 = []
        fc2.append(EqualizedLinear(1024, 512))
        fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc2.append(EqualizedLinear(512, 512))
        fc2.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
        fc2.append(EqualizedLinear(512, key_size))
        self.fc1 = nn.Sequential(*fc1)
        self.fc2 = nn.Sequential(*fc2)

    def forward(self, x, m):
        x = self.model(x)
        m = self.fc1(m)
        x = self.fc2(torch.cat([x, m], 1))
        # 删除sigmoid？
        x = F.sigmoid(x)
        return x


class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    # channel 64 | num_blocks 4
    def __init__(self, num_blocks, channels, scaling_i=1.0, scaling_w=0.2,last_tanh=True):
        super(HiddenEncoder, self).__init__()
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs):
        encoded_image = self.conv_bns(imgs)  # b c h w
        im_w = self.final_layer(encoded_image)
        if self.last_tanh:
            im_w = self.tanh(im_w)
        imgs = self.scaling_i * imgs + self.scaling_w * im_w
        return imgs


class HiddenEncoderLatent(nn.Module):
    """
    Inserts a watermark into an image.
    """
    # channel 64 | num_blocks 4
    def __init__(self, num_blocks, channels, scaling_i=1.0, scaling_w=0.2,last_tanh=True):
        super(HiddenEncoderLatent, self).__init__()
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        layers = [ConvBNRelu(4, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(channels, 4, kernel_size=1)
        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs):
        encoded_image = self.conv_bns(imgs)  # b c h w
        im_w = self.final_layer(encoded_image)
        if self.last_tanh:
            im_w = self.tanh(im_w)
        imgs = self.scaling_i * imgs + self.scaling_w * im_w
        return imgs