import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_Res(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


class Up_Res(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # //为整数除法

        self.conv = DoubleConv_Res(in_channels, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # //为整数除法

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Upp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # //为整数除法

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, scale_i=0.9, scale_w=0.1, bilinear=True):
        super(UNet, self).__init__()
        self.scale_i = scale_i
        self.scale_w = scale_w
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        self.bilinear = bilinear  # 上采样方式

        self.inc = DoubleConv(n_channels, 64)  # 输入层
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)  # 输出层

    def forward(self, x):
        inp = x
        x1 = self.inc(x)  # 一开始输入
        x2 = self.down1(x1)  # 四层左部分
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)  # 四层右部分
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # 最终输出
        logits = inp * self.scale_i + logits * self.scale_w
        return logits


class UNet_1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_1, self).__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        self.bilinear = bilinear  # 上采样方式

        self.inc = DoubleConv(n_channels, 64)  # 输入层
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Upp(512, 256, bilinear)
        self.up2 = Upp(256, 128, bilinear)
        self.up3 = Upp(128, 64, bilinear)
        self.up4 = Upp(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)  # 输出层
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

    def forward(self, x):
        x1 = self.inc(x)  # 一开始输入
        x2 = self.down1(x1)  # 四层左部分
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.sca(x5) * x5
        x = self.up1(x5)  # 四层右部分
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)  # 最终输出
        return logits


class UNet_1_Encoder(nn.Module):
    def __init__(self, n_channels):
        super(UNet_1_Encoder, self).__init__()
        self.n_channels = n_channels  # 输入通道数

        self.inc = DoubleConv(n_channels, 64)  # 输入层
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)  # 2048

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.sca(x) * x
        return x


class Classify(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(Classify, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(x)


class Projection(nn.Module):
    def __init__(self, prev_dim, dim=2048):
        super(Projection, self).__init__()
        self.pjo = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                 nn.BatchNorm1d(prev_dim),
                                 nn.ReLU(inplace=True),  # first layer
                                 nn.Linear(prev_dim, prev_dim, bias=False),
                                 nn.BatchNorm1d(prev_dim),
                                 nn.ReLU(inplace=True),  # second layer
                                 nn.Linear(prev_dim, dim, bias=False),
                                 nn.BatchNorm1d(dim, affine=False))  # output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.pjo(x)


class Predictor(nn.Module):
    def __init__(self, dim, pred_dim=512):
        super(Predictor, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(x)


class MLP(nn.Module):
    def __init__(self, feat_dim):
        super(MLP, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(x)


class UNet_1_Decoder(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(UNet_1_Decoder, self).__init__()
        self.n_classes = n_classes  # 输出类别数
        self.bilinear = bilinear  # 上采样方式

        self.up1 = Upp(512, 256, bilinear)
        self.up2 = Upp(256, 128, bilinear)
        self.up3 = Upp(128, 64, bilinear)
        self.up4 = Upp(64, 64, bilinear)

        self.outc = OutConv(64, n_classes)  # 输出层)

    def forward(self, x):
        x = self.up1(x)  # 四层右部分
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)  # 最终输出
        return logits


class UNet_1_Decoder_Res(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(UNet_1_Decoder_Res, self).__init__()
        self.n_classes = n_classes  # 输出类别数
        self.bilinear = bilinear  # 上采样方式

        self.up1 = Up_Res(512, 256, bilinear)
        self.up2 = Up_Res(256, 128, bilinear)
        self.up3 = Up_Res(128, 64, bilinear)
        self.up4 = Up_Res(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)  # 输出层

    def forward(self, x):

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)  # 最终输出
        return logits


class DeBlurNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeBlurNet, self).__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        self.bilinear = bilinear  # 上采样方式

        self.inc = DoubleConv(n_channels, 64)  # 输入层
        self.down1 = Down(64, 64)
        self.up1 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)  # 输出层

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.up1(x2, x1)
        logits = self.outc(x3) + x
        return logits


def get_model():
    model = UNet(3, 3)
    return model
