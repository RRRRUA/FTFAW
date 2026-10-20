import sys
from sklearn import metrics
import cv2
import torch
from imwatermark import WatermarkEncoder, WatermarkDecoder
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import lpips
from torchvision.utils import save_image
from image_augmentation import TestDataAugmentation, fill_image_single

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1],
                                         std=[1 / 0.5, 1 / 0.5, 1 / 0.5])  # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225])  # Unnormalize (x * std) + mean


def psnr(x, y, img_space='vqgan'):
    """
    Return PSNR
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20 * np.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))  # B
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


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


def gen_key_list(key_size, key_num):
    file = 'key_list.txt'
    with open(file, "w", encoding='utf-8') as f:
        for i in range(int(key_num)):
            random_array = np.random.randint(0, 2, size=(key_size,))
            str = np.array2string(random_array, separator='', max_line_width=key_size + 2)
            str = str[1:-1] + "\n"
            f.writelines(str)
    return


def get_key_list(key_num):
    file = 'key_list.txt'
    key_list_tensor = torch.zeros(1, 48).to(device)
    with open(file, "r", encoding='utf-8') as f:
        for i in range(int(key_num)):
            str = f.readline()[:-1]
            key_tensor = torch.tensor([float(str[j]) for j in range(len(str))]).to(device)
            key_list_tensor = torch.cat([key_list_tensor, key_tensor.unsqueeze(0)], dim=0)
    return key_list_tensor[1:]


# 后处理水印嵌入
def watermark_encode(key_list, num, watermark_method, orig_path):
    # dwtDctsvd dwtDct rivaGan(只支持32bit)
    images_path = orig_path + '/'
    watermark_images_path = orig_path + '_' +  watermark_method + '/'
    print('orig_path: ', images_path)
    print('watermark_images_path: ', watermark_images_path)
    print('start encoding>>>')
    for i in range(num):
        bgr = cv2.imread(images_path + f'{i}.png')
        wm = key_list[i]
        encoder = WatermarkEncoder()
        encoder.set_watermark('bits', wm)
        bgr_encoded = encoder.encode(bgr, watermark_method)
        cv2.imwrite(watermark_images_path + f'{i}.png', bgr_encoded)
    print('finish')


def watermark_decode(key_list, num, key_size, watermark_method, orig_path):
    watermark_images_path = orig_path + '_' +  watermark_method + '/'
    orig_images_path = orig_path + '/'
    decoder = WatermarkDecoder('bits', key_size)
    bit_acc = []
    img_size = 512
    aug = TestDataAugmentation(img_size=img_size, device=device).to(device)
    bit_acc_orig = []
    bit_acc_w = []
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
    for i in range(len(attacks)):
        bit_acc_w.append([])
        bit_acc.append(0)
    print('attack_num: ', len(attacks))
    print('orig_path: ', orig_images_path)
    print('watermark_images_path: ', watermark_images_path)
    print('start decoding>>>')
    for i in range(num):
        key = key_list[i].cuda()
        bgr = cv2.imread(watermark_images_path + f'{i}.png')
        bgr_orig = cv2.imread(orig_images_path + f'{i}.png')

        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img_tensor = torch.tensor(bgr, dtype=torch.float64).cuda() / 255
        j = 0
        for name, attack in attacks.items():
            img_att = attack(img_tensor)
            if 'crop' in name:
                img_att = fill_image_single(img_att, (3, img_size, img_size))
            # save_image(torch.clamp(img_att, 0, 1), f'{name}_test.png', nrow=8)
            bgr = np.uint8(img_att.cpu().numpy() * 255)
            bgr = bgr.transpose(1, 2, 0)
            bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            ext_key = decoder.decode(bgr, watermark_method)
            k = torch.abs(key - torch.tensor(ext_key).float().to(device)).sum()
            acc = (key_size - k) / key_size
            bit_acc[j] += acc
            bit_acc_w[j].append(acc.cpu())
            j += 1
            if 'none' in name:
                ext_key_orig = decoder.decode(bgr_orig, watermark_method)
                k = torch.abs(key - torch.tensor(ext_key_orig).float().to(device)).sum()
                acc = (key_size - k) / key_size
                bit_acc_orig.append(acc.cpu())

    i = 0
    print(watermark_method)
    for name, attack in attacks.items():
        auc, acc, low = get_roc(bit_acc_orig, bit_acc_w[i])
        print(name, ' auc:{:.4f}|acc:{:.4f}|TPR@0.01FPR:{:.4f}'.format(auc, acc, low))
        print(name, ' avg_bit_acc: ', bit_acc[i] / num)
        i += 1


@torch.no_grad()
def image_quality(image_path1, image_path2):
    vqgan_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_vqgan,
    ])
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    num = 5000
    avg_psnr = 0
    avg_psnr_t = 0
    avg_ssim = 0
    avg_ssim_t = 0
    avg_lpips = 0
    avg_lpips_t = 0
    for i in range(num):
        image1_orig = Image.open(f'{image_path1}/{i}.png')
        image2_orig = Image.open(f'{image_path2}/{i}.png')
        image1, image2 = vqgan_transform(image1_orig).cuda(), vqgan_transform(image2_orig).cuda()
        image1_t, image2_t = tensor_transform(image1_orig).cuda(), tensor_transform(image2_orig).cuda()
        avg_psnr_t += psnr(image1_t.unsqueeze(0), image2_t.unsqueeze(0))
        avg_psnr += psnr(image1.unsqueeze(0), image2.unsqueeze(0))
        avg_ssim += ssim(image1.unsqueeze(0), image2.unsqueeze(0))
        avg_ssim_t += ssim(image1_t.unsqueeze(0), image2_t.unsqueeze(0))
        avg_lpips_t += loss_fn_vgg.forward(image1_t, image2_t).mean()
        avg_lpips += loss_fn_vgg.forward(image1, image2).mean()
    print('avg_psnr: ', avg_psnr / num)
    print('avg_psnr_t: ', avg_psnr_t / num)
    print('avg_ssim: ', avg_ssim / num)
    print('avg_ssim_t: ', avg_ssim_t / num)
    print('avg_lpips: ', avg_lpips / num)
    print('avg_lpips_t: ', avg_lpips_t / num)


if __name__ == '__main__':
    # mscoco
    # dwtDct_48 | avg_bit_acc: 0.8826  | psnr: 37.1191 | ssim: 0.9488 | adv_bit_acc: 0.5881 | TPR: | TPR(adv)： | bit acc(adv): 0.5881
    # dwtDctSvd_48 | avg_bit_acc: 0.9998 | psnr: 37.2084  | ssim: 0.9538 | adv_bit_acc: 0.6134 | TPR：| TPR(adv)： | bit acc(adv): 0.6136

    # laion
    # dwtDct_48 | avg_bit_acc:   | psnr:  | ssim:  | adv_bit_acc:  | TPR: | TPR(adv)： | bit acc(adv): 
    # dwtDctSvd_48 | avg_bit_acc:  | psnr:   | ssim:  | adv_bit_acc:  | TPR：| TPR(adv)： | bit acc(adv): 

    num = 5000
    method = 'dwtDct'
    key_size = 48
    key_list = get_key_list(key_num=num)
    orig_path = '/media/wang004/data/panjiajun/data/image_samples/sd2-1-base_sample_orig_DPMSolver'
    method_list = ['dwtDct', 'dwtDctSvd']
    for me in method_list:
        # watermark_encode(key_list, num=num, watermark_method=me, orig_path=orig_path)
        # watermark_decode(key_list, num=num, key_size=key_size, watermark_method=me, orig_path=orig_path)
        image_quality(f'/media/wang004/data/panjiajun/data/image_samples/sd2-1-base_sample_orig_DPMSolver_{me}',
        '/media/wang004/data/panjiajun/data/image_samples/sd2-1-base_sample_orig_DPMSolver')
        print(me)
    '''
    image_quality('/media/wang004/data/panjiajun/data/image_samples/sd_sample_orig', 
    '/media/wang004/data/panjiajun/data/image_samples/sd_sample_stable_signature_1')
    '''
