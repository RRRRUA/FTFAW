import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EPS = 1e-10


# 获取vgg特征提取模块其中的5个输出
class VggFeatureExtractor(nn.Module):
    def __init__(self):
        super(VggFeatureExtractor, self).__init__()

        # download vgg 读取预训练模型
        vgg16 = torchvision.models.vgg16(pretrained=True).features

        # set non trainable
        for param in vgg16.parameters():
            param.requires_grad = False

        # slice model
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # 将vgg的特征提取模块分为5块
        for x in range(4):  # conv relu conv relu
            self.slice1.add_module(str(x), vgg16[x])
        for x in range(4, 9):  # max conv relu conv relu
            self.slice2.add_module(str(x), vgg16[x])
        for x in range(9, 16):  # max cov relu conv relu conv relu
            self.slice3.add_module(str(x), vgg16[x])
        for x in range(16, 23):  # conv relu max conv relu conv relu
            self.slice4.add_module(str(x), vgg16[x])
        for x in range(23, 30):  # conv relu conv relu max conv relu
            self.slice5.add_module(str(x), vgg16[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


def normalize_tensor(t):
    # norms a tensor over the channel dimension to an euclidean length of 1.
    # 将通道维度上的张量规范为欧几里得长度1
    N, C, H, W = t.shape
    norm_factor = torch.sqrt(torch.sum(t ** 2, dim=1)).view(N, 1, H, W)
    return t / (norm_factor.expand_as(t) + EPS)


def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:, :, :, :, 0] + b * softmax_factors[:, :, :, :, 1]


class WatsonDistanceVgg(nn.Module):
    """
    Loss function based on Watsons perceptual distance.
    Based on deep feature extraction
    """

    def __init__(self, trainable=False, reduction='sum'):
        """
        Parameters:
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()

        # module to perform feature extraction
        self.add_module('vgg', VggFeatureExtractor())

        # imagenet-normalization
        self.shift = nn.Parameter(torch.Tensor([-.030, -.088, -.188]).view(1, 3, 1, 1), requires_grad=False)
        self.scale = nn.Parameter(torch.Tensor([.458, .448, .450]).view(1, 3, 1, 1), requires_grad=False)

        # channel dimensions
        self.L = 5
        self.channels = [64, 128, 256, 512, 512]

        # sensitivity parameters
        self.t0_tild = nn.Parameter(torch.zeros((self.channels[0])), requires_grad=trainable)
        self.t1_tild = nn.Parameter(torch.zeros((self.channels[1])), requires_grad=trainable)
        self.t2_tild = nn.Parameter(torch.zeros((self.channels[2])), requires_grad=trainable)
        self.t3_tild = nn.Parameter(torch.zeros((self.channels[3])), requires_grad=trainable)
        self.t4_tild = nn.Parameter(torch.zeros((self.channels[4])), requires_grad=trainable)

        # other default parameters
        w = torch.tensor(0.2)  # contrast masking
        self.w0_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)  # inverse of sigmoid
        self.w1_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.w2_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.w3_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.w4_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable)  # pooling

        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)

        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        return [torch.exp(t) for t in [self.t0_tild, self.t1_tild, self.t2_tild, self.t3_tild, self.t4_tild]]

    @property
    def w(self):
        # return luminance masking parameter
        return [torch.sigmoid(w) for w in [self.w0_tild, self.w1_tild, self.w2_tild, self.w3_tild, self.w4_tild]]

    def forward(self, input, target):
        # normalization
        # expand_as：输入tensor的维度扩展为与指定tensor相同的size
        # 将输入图像根据imagenet的均值和方差进行归一化
        input = (input - self.shift.expand_as(input)) / self.scale.expand_as(input)
        target = (target - self.shift.expand_as(target)) / self.scale.expand_as(target)

        # 获取特征图
        c0 = self.vgg(target)
        c1 = self.vgg(input)

        # norm over channels
        for l in range(self.L):
            c0[l] = normalize_tensor(c0[l])
            c1[l] = normalize_tensor(c1[l])

        # contrast masking
        t = self.t
        w = self.w
        s = []
        for l in range(self.L):
            N, C_l, H_l, W_l = c0[l].shape
            t_l = t[l].view(1, C_l, 1, 1).expand(N, C_l, H_l, W_l)
            s.append(softmax(t_l, (c0[l].abs() + EPS) ** w[l] * t_l ** (1 - w[l])))

        # pooling
        watson_dist = 0
        for l in range(self.L):
            _, _, H_l, W_l = c0[l].shape
            layer_dist = (((c0[l] - c1[l]) / s[l]).abs() + EPS) ** self.beta
            layer_dist = self.dropout(layer_dist) + EPS
            layer_dist = torch.sum(layer_dist, dim=(1, 2, 3))  # sum over dimensions of layer
            layer_dist = (1 / (H_l * W_l)) * layer_dist  # normalize by layer size
            watson_dist += layer_dist  # sum over layers
        watson_dist = watson_dist ** (1 / self.beta)

        # reduction
        if self.reduction == 'sum':
            watson_dist = torch.sum(watson_dist)

        return watson_dist


class FocalWatsonDistanceVgg(nn.Module):
    """
    Loss function based on Watsons perceptual distance.
    Based on deep feature extraction
    """

    def __init__(self, trainable=False, reduction='sum'):
        """
        Parameters:
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()

        # module to perform feature extraction
        self.add_module('vgg', VggFeatureExtractor())

        # imagenet-normalization
        self.shift = nn.Parameter(torch.Tensor([-.030, -.088, -.188]).view(1, 3, 1, 1), requires_grad=False)
        self.scale = nn.Parameter(torch.Tensor([.458, .448, .450]).view(1, 3, 1, 1), requires_grad=False)

        # channel dimensions
        self.L = 5
        self.channels = [64, 128, 256, 512, 512]

        # sensitivity parameters
        self.t0_tild = nn.Parameter(torch.zeros((self.channels[0])), requires_grad=trainable)
        self.t1_tild = nn.Parameter(torch.zeros((self.channels[1])), requires_grad=trainable)
        self.t2_tild = nn.Parameter(torch.zeros((self.channels[2])), requires_grad=trainable)
        self.t3_tild = nn.Parameter(torch.zeros((self.channels[3])), requires_grad=trainable)
        self.t4_tild = nn.Parameter(torch.zeros((self.channels[4])), requires_grad=trainable)

        # other default parameters
        w = torch.tensor(0.2)  # contrast masking
        self.w0_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)  # inverse of sigmoid
        self.w1_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.w2_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.w3_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.w4_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable)  # pooling

        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)

        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        return [torch.exp(t) for t in [self.t0_tild, self.t1_tild, self.t2_tild, self.t3_tild, self.t4_tild]]

    @property
    def w(self):
        # return luminance masking parameter
        return [torch.sigmoid(w) for w in [self.w0_tild, self.w1_tild, self.w2_tild, self.w3_tild, self.w4_tild]]

    def forward(self, input, target):
        # normalization
        # expand_as：输入tensor的维度扩展为与指定tensor相同的size
        # 将输入图像根据imagenet的均值和方差进行归一化
        input = (input - self.shift.expand_as(input)) / self.scale.expand_as(input)
        target = (target - self.shift.expand_as(target)) / self.scale.expand_as(target)

        # 获取特征图
        c0 = self.vgg(target)
        c1 = self.vgg(input)

        # norm over channels
        for l in range(self.L):
            c0[l] = normalize_tensor(c0[l])
            c1[l] = normalize_tensor(c1[l])

        # contrast masking
        t = self.t
        w = self.w
        s = []
        for l in range(self.L):
            N, C_l, H_l, W_l = c0[l].shape
            t_l = t[l].view(1, C_l, 1, 1).expand(N, C_l, H_l, W_l)
            s.append(softmax(t_l, (c0[l].abs() + EPS) ** w[l] * t_l ** (1 - w[l])))

        # pooling
        watson_dist = 0
        # watson_dist = []
        for l in range(self.L):
            _, _, H_l, W_l = c0[l].shape
            layer_dist = (((c0[l] - c1[l]) / s[l]).abs() + EPS) ** self.beta
            layer_dist = self.dropout(layer_dist) + EPS
            # 计算权重损失
            layer_dist = get_focal_feature(layer_dist)
            # layer_dist = torch.sum(layer_dist, dim=(2, 3))
            # layer_dist = get_focal_channels(layer_dist)
            # layer_dist = torch.sum(layer_dist, dim=(1,))
            layer_dist = torch.sum(layer_dist, dim=(1, 2, 3))  # sum over dimensions of layer
            layer_dist = (1 / (H_l * W_l)) * layer_dist  # normalize by layer size 减去特征图尺寸对损失值的影响
            watson_dist += layer_dist  # sum over layers
            # watson_dist.append(layer_dist)
        # 为不同特征图添加权重
        watson_dist = watson_dist ** (1 / self.beta)

        # reduction
        if self.reduction == 'sum':
            watson_dist = torch.sum(watson_dist)
        
        return watson_dist


# 根据特征图的差值，计算焦点值
def get_focal_feature(x, batch_matrix=False):
    # 此处应该开根号
    matrix_tmp = torch.sqrt(x ** 2)
    # matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, None, None]
    # 使用指数计算可以缩小权重的差异
    matrix_tmp = torch.exp(matrix_tmp) / torch.exp(matrix_tmp.max(-1).values.max(-1).values[:, :, None, None])
    matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
    matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
    weight = matrix_tmp.clone().detach()
    feature = weight * x
    return feature


# 为所有特征层添加权重（需要判断是否有必要，以及指数加权）
# 感知损失中，浅层特征表示基础的信息，深层特征表示抽象信息
# 训练后期，差异主要表现在深层特征，因此使用权重会削弱浅层特征差异的重要性
# 为不同的通道加权
def get_focal_channels(layer_dist):
    factor = layer_dist.clone()
    for i in range(factor.size(0)):
        factor[i] = factor[i] / torch.max(factor[i])
    factor[torch.isnan(factor)] = 0.0
    factor = torch.clamp(factor, min=0.0, max=1.0)
    factor = factor.detach()
    layer_dist = factor * layer_dist
    return layer_dist
