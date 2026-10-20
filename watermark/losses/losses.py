import torch
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
import torch.nn as nn



# 计算预测的噪声
def get_noise(model,
              x0: torch.Tensor,
              t: torch.LongTensor,
              e: torch.Tensor,
              b: torch.Tensor,
              w: torch.Tensor):
    # 根据采样的时间表t获得a的累乘表
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # 根据原始图像x0和噪声e，使用公式计算xt
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # 将xt和时间表输入模型，得到预测的噪声
    output = model(x, t.float(), w)
    return output


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          keepdim=False):
    # 根据采样的时间表t获得a的累乘表
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # 根据原始图像x0和噪声e，使用公式计算xt
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    # 将xt和时间表和掩模后的水印输入模型，得到预测的噪声
    output = model(x, t.float())
    # 计算预测噪声和真实噪声的差平方的和作为损失值
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# 已知预测噪声时计算预测损失
def estimation_loss(e: torch.Tensor,
                    output: torch.Tensor, keepdim=False):
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


# 计算不同密钥时的预测噪声的一致性损失,使用L2范数的平方,计算均值
def consistency_loss(output1: torch.Tensor,
                     output2: torch.Tensor):
    return torch.dist(output1, output2, p=2) ** 2 / output1.size(0)


# 计算提取的密钥和原始密钥之间的BCE损失，计算均值
def key_extract_loss(extract_key: torch.Tensor,
                     real_key: torch.Tensor):
    real_key = real_key.repeat(extract_key.size(0))
    extract_key = extract_key.flatten()
    bce_loss = torch.nn.BCELoss(reduction='mean')
    return bce_loss(extract_key, real_key)


# 计算水印掩模的L1损失，令嵌入的水印位置尽可能少（希望增加对微调的鲁棒性）
def mask_loss_orig(m):
    return m.sum()


# 计算掩模率，作为损失值
def mask_loss(m):
    return m.sum() / m.size(0)


# 计算位精度
def get_bit_acc(extract_key: torch.Tensor,
                real_key: torch.Tensor):
    real_key = real_key.repeat(extract_key.size(0))
    extract_key = torch.round(extract_key.reshape(-1))
    real_key = real_key.reshape(-1)
    k = torch.abs(real_key - extract_key).sum()
    acc = (real_key.size(0) - k) / real_key.size(0)
    return acc


def get_bit_acc_list(extract_key, real_key):
    list = []
    real_key = real_key.unsqueeze(0).repeat(extract_key.size(0), 1)
    extract_key = torch.round(extract_key)
    k = torch.sum(torch.abs(real_key - extract_key), dim=1)
    acc = (real_key.size(1) - k) / real_key.size(1)
    for i in range(acc.size(0)):
       list.append(acc[i].item())
    return list


def get_roc(acc_nw, acc_w):
    preds = acc_nw + acc_w
    t_labels = [0] * len(acc_nw) + [1] * len(acc_w)
    # 设置阳性标签为1，计算不同阈值时的真假阳性率
    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # 计算最大的归因精度
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    # 计算假阳性率为0.01（1%）时的真阳性率
    low = tpr[np.where(fpr < 0.01)[0][-1]]
    return auc, acc, low
 
 
 # 计算mask的平均重合率
def get_mask_coincidence_rate(mask, mask_temp):
    coincidence_rate = (mask.size(0) - torch.abs(mask - mask_temp).sum()) / mask.size(0)
    return coincidence_rate


def get_bit_acc_stable_signature(extract_key, real_key):
    real_key = real_key.repeat(extract_key.size(0))
    extract_key = extract_key.reshape(-1)
    real_key = real_key.reshape(-1)
    k = (~torch.logical_xor(extract_key > 0, real_key > 0))
    acc = torch.sum(k, -1) / real_key.size(0)
    return acc


def get_bit_acc_list_stable_signature(extract_key, real_key):
    list = []
    real_key = real_key.unsqueeze(0).repeat(extract_key.size(0), 1)
    extract_key = extract_key
    k = (~torch.logical_xor(extract_key > 0, real_key > 0))
    acc = torch.sum(k, -1) / real_key.size(0)
    for i in range(acc.size(0)):
       list.append(acc[i].item())
    return list


def key_extract_loss_stable_signature(extract_key: torch.Tensor,
                     real_key: torch.Tensor):
    real_key = real_key.repeat(extract_key.size(0))
    extract_key = extract_key.flatten()
    temp = 10
    loss = F.binary_cross_entropy_with_logits(extract_key * temp, real_key, reduction='mean')
    return loss


def key_extract_loss_mask(extract_key: torch.Tensor,
                     real_key: torch.Tensor):
    real_key = real_key.flatten()
    extract_key = extract_key.flatten()
    bce_loss = torch.nn.BCELoss(reduction='mean')
    return bce_loss(extract_key, real_key)


def get_bit_acc_mask(extract_key, real_key):
    extract_key = torch.round(extract_key.reshape(-1))
    real_key = real_key.reshape(-1)
    k = torch.abs(real_key - extract_key).sum()
    acc = (real_key.size(0) - k) / real_key.size(0)
    return acc


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        pred = pred * 255
        target = target * 255
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()



loss_registry = {
    'get_noise': get_noise,
    'simple': noise_estimation_loss,
    'consistence_loss': consistency_loss,
    'estimation_loss': estimation_loss,
    'key_extract_loss': key_extract_loss,
    'mask_loss': mask_loss
}
