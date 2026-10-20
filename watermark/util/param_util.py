import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 获取模型卷积层的参数
def get_param(model):
    conv_param = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            conv_param.append(layer.weight)
    return conv_param


# 获得模型中卷积层的数量以及各自的通道数,并计算总通道数
def get_param_size_all(model):
    channels = []
    channels_size = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            channels.append((layer.weight.size(0), layer.weight.size(1)))
            channels_size += layer.weight.size(0) * layer.weight.size(1)
    return channels, channels_size


def get_param_size(model):
    channels = []
    channels_size = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            channels.append(layer.weight.size(0))
            channels_size += layer.weight.size(0)
    return channels, channels_size


def get_param_size_wouaf(model):
    channels = []
    channels_size = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            channels.append(layer.weight.size(1))
            channels_size += layer.weight.size(1)
    return channels, channels_size


def copy_param(model1, model2):
    paras = []
    for para in model1.parameters():
        paras.append(para.data)
    i = 0
    for para in model2.parameters():
        para.data.copy_(paras[i])
        i += 1
    i = 0
    for para in model2.parameters():
        if not torch.equal(para.data, paras[i]):
            print("unequal")
        i += 1
    return model2


def equal_param(model1, model2):
    paras = []
    for para in model1.parameters():
        paras.append(para.data)
    i = 0
    for para in model2.parameters():
        if not torch.equal(para.data, paras[i]):
            print("unequal")
        i += 1
    return


def get_param_name_size(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(name)
            print(layer.weight.size())
    return 0


def modify_param(model, conv_param, w, m):
    w[m == 0] = 1
    k = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.weight = torch.nn.Parameter(conv_param[k] * w)
            k += 1
    return model
