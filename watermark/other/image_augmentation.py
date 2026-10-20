import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as k

noise_list = [k.RandomGaussianNoise(std=0.05, keepdim=True, p=0.5),
              k.RandomGaussianNoise(std=0.1, keepdim=True, p=0.5),
              k.RandomGaussianNoise(std=0.15, keepdim=True, p=0.5),
              k.RandomGaussianNoise(std=0.2, keepdim=True, p=0.5)]


# 难度较高的增强（旋转，模糊，jpeg），是否应该放在队列的前位？
def fill_image(image, img_size):
    if img_size[1] != image.size(2) or img_size[2] != image.size(3):
        left = int((img_size[2] - image.size(3)) / 2)
        right = img_size[2] - image.size(3) - left
        top = int((img_size[1] - image.size(2)) / 2)
        bottom = img_size[1] - image.size(2) - top
        p2d = (left, right, top, bottom)
        image = F.pad(image, p2d, 'constant', 0)
    return image


def fill_image_single(image, img_size):
    if img_size[1] != image.size(1) or img_size[2] != image.size(2):
        left = int((img_size[2] - image.size(2)) / 2)
        right = img_size[2] - image.size(2) - left
        top = int((img_size[1] - image.size(1)) / 2)
        bottom = img_size[1] - image.size(1) - top
        p2d = (left, right, top, bottom)
        image = F.pad(image, p2d, 'constant', 0)
    return image


def aug_jpeg(level):
    strength = (100.0 - level * 10, 90.0)
    return k.RandomJPEG(jpeg_quality=strength, keepdim=True, p=0.5)


def aug_erasing(level):
    strength = (0.05, 0.05 * (level if level < 5 else 4))
    return k.RandomErasing(scale=strength, keepdim=True, p=0.5)


def aug_rotation(level):
    strength = (-5 - 5 * level, 5 + 5 * level)
    return k.RandomRotation(degrees=strength, keepdim=True, p=0.5)


def aug_gaussianblur(level):
    kernel_size_list = []
    kernel_size_list.append((3, 3))
    if level >= 3:
        kernel_size_list.append((5, 5))
        if level == 5:
            kernel_size_list.append((7, 7))
    return k.RandomGaussianBlur(kernel_size=random.choice(kernel_size_list), sigma=(1.0, 1.0), keepdim=True, p=0.5)


def aug_crop(level, crop_list):
    seq = k.AugmentationSequential(
        *crop_list[:level],
        random_apply=1
    )
    return seq


def aug_brightness(level):
    strength = (0.95 - 0.05 * level, 1.05 + 0.05 * level)
    return k.RandomBrightness(brightness=strength, keepdim=True, p=0.5)


def aug_noise(level):
    seq = k.AugmentationSequential(
        *noise_list[:(level if level < 5 else -1)],
        random_apply=1
    )
    return seq


# 训练时数据增强，使用不同增强的随机组合，增强的类型和强度来自WOUAF
# wouaf使用了深度压缩模型作为数据增强，模型来自compressai
class TrainDataAugmentation(nn.Module):
    def __init__(self, img_size):
        super(TrainDataAugmentation, self).__init__()
        self.num = 1
        self.level = 1
        self.crop_list = [k.RandomCrop(size=(img_size, img_size), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.95), int(img_size * 0.95)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.9), int(img_size * 0.9)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.85), int(img_size * 0.85)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.8), int(img_size * 0.8)), keepdim=True, p=0.5)]

    def forward(self, x, num):
        if num > self.num:
            self.num += 1
            self.level = self.level + 1 if self.level < 5 else 1
        aug_list = []
        if num > 0:
            aug_list.append(aug_noise(self.level if num < 6 else 5))
        if num > 5:
            # wouaf使用的jpeg压缩为diffjpeg，可微分的jpeg压缩（kornia中由diffjpeg）
            aug_list.append(aug_jpeg(self.level if num < 11 else 5))
        if num > 10:
            aug_list.append(aug_rotation(self.level if num < 16 else 5))
        if num > 15:
            aug_list.append(aug_gaussianblur(self.level if num < 21 else 5))
        if num > 20:
            aug_list.append(aug_crop(self.level if num < 26 else 5, crop_list=self.crop_list))
        if num > 25:
            aug_list.append(aug_brightness(self.level if num < 31 else 5))
        if num > 30:
            aug_list.append(aug_erasing(self.level if num < 36 else 5))
        aug = k.AugmentationSequential(
            *aug_list,
            random_apply=False
        )
        x = aug(x)
        return x


# 同时加入所有数据增强，强度随机
class TrainDataAugmentationAll(nn.Module):
    def __init__(self, img_size):
        super(TrainDataAugmentationAll, self).__init__()
        self.num = 1
        self.level = 5
        self.crop_list = [k.RandomCrop(size=(img_size, img_size), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.95), int(img_size * 0.95)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.9), int(img_size * 0.9)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.85), int(img_size * 0.85)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.8), int(img_size * 0.8)), keepdim=True, p=0.5)]

    def forward(self, x, num=None):
        aug_list = []
        aug_list.append(aug_noise(self.level))
        aug_list.append(aug_jpeg(self.level))
        aug_list.append(aug_rotation(self.level))
        aug_list.append(aug_gaussianblur(self.level))
        aug_list.append(aug_crop(self.level, crop_list=self.crop_list))
        aug_list.append(aug_brightness(self.level))
        aug_list.append(aug_erasing(self.level))
        aug = k.AugmentationSequential(
            *aug_list,
            random_apply=False
        )
        x = aug(x)
        return x


# 同时加入所有数据增强，强度渐进
class TrainDataAugmentationAllGradual(nn.Module):
    def __init__(self, img_size):
        super(TrainDataAugmentationAllGradual, self).__init__()
        self.num = 1
        self.level = 5
        self.crop_list = [k.RandomCrop(size=(img_size, img_size), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.95), int(img_size * 0.95)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.9), int(img_size * 0.9)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.85), int(img_size * 0.85)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.8), int(img_size * 0.8)), keepdim=True, p=0.5)]

    def forward(self, x, level=1):
        # 强度渐进，当前强度时的bit_acc达到0.9的次数到10时，level+1
        aug_list = []
        aug_list.append(aug_noise(level))
        aug_list.append(aug_jpeg(level))
        aug_list.append(aug_rotation(level))
        aug_list.append(aug_gaussianblur(level))
        aug_list.append(aug_crop(level, crop_list=self.crop_list))
        aug_list.append(aug_brightness(level))
        aug_list.append(aug_erasing(level))
        aug = k.AugmentationSequential(
            *aug_list,
            random_apply=False
        )
        x = aug(x)
        return x


# 测试时图像增强，根据给定的类型和对应的强度选择
class TestDataAugmentation(nn.Module):
    def __init__(self, img_size, device):
        super(TestDataAugmentation, self).__init__()
        self.img_size = img_size
        self.device = device
        self.crop_list = [k.RandomCrop(size=(img_size, img_size), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.95), int(img_size * 0.95)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.9), int(img_size * 0.9)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.85), int(img_size * 0.85)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.8), int(img_size * 0.8)), keepdim=True, p=0.5)]
        # 不同增强的组合
        self.combine = k.AugmentationSequential(
            # 随机JPEG压缩
            k.RandomJPEG(jpeg_quality=(50.0, 90.0), keepdim=True, p=0.5),
            # 随机擦除
            k.RandomErasing(scale=(0.05, 0.2), keepdim=True, p=0.5),
            # 随机旋转
            k.RandomRotation(degrees=(-30, 30), keepdim=True, p=0.5),
            # 随机高斯模糊
            k.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.0, 1.0), keepdim=True, p=0.5),
            # 随机裁剪[0.8-1.0]
            aug_crop(5, crop_list=self.crop_list),
            # 随机亮度调整
            k.RandomBrightness(brightness=(0.7, 1.3), keepdim=True, p=0.5),
            # 随机高斯噪声(0.05-0.2)
            k.AugmentationSequential(
                k.RandomGaussianNoise(std=0.05, keepdim=True, p=0.5),
                k.RandomGaussianNoise(std=0.1, keepdim=True, p=0.5),
                k.RandomGaussianNoise(std=0.15, keepdim=True, p=0.5),
                k.RandomGaussianNoise(std=0.2, keepdim=True, p=0.5),
                random_apply=1
            ),
            # 按照原始顺序依次按概率使用图像增强
            random_apply=False,
        )

    def forward(self, aug_type, strength, x):
        aug = k.AugmentationSequential().to(self.device)
        if aug_type == 'Erasing':
            x = k.RandomErasing(scale=strength, keepdim=True, p=1)(x)
        if aug_type == 'Rotation':
            x = k.RandomRotation(degrees=strength, keepdim=True, p=1)(x)
        if aug_type == 'Blur':
            x = k.RandomGaussianBlur(kernel_size=strength, sigma=(1.0, 1.0), keepdim=True, p=1)(x)
        if aug_type == 'Crop':
            x = k.RandomCrop(size=strength, keepdim=True, p=1)(x)
        if aug_type == 'Brightness':
            x = k.RandomBrightness(brightness=strength, keepdim=True, p=1)(x)
        if aug_type == 'Noise':
            x = k.RandomGaussianNoise(std=strength, keepdim=True, p=1)(x)
        if aug_type == 'JPEG':
            x = k.RandomJPEG(jpeg_quality=strength, keepdim=True, p=1)(x)
        if aug_type == 'Combine':
            # wouaf中的设置为p=1
            pp = 1
            combine = k.AugmentationSequential(
                # 随机JPEG压缩
                k.RandomJPEG(jpeg_quality=(100.0 - strength * 10, 100.0 - strength * 10), keepdim=True, p=pp),
                # 随机擦除
                k.RandomErasing(
                    scale=(0.05 * (strength if strength < 5 else 4), 0.05 * (strength if strength < 5 else 4)),
                    keepdim=True, p=pp),
                # 随机旋转
                k.RandomRotation(degrees=(5 + 5 * strength, 5 + 5 * strength), keepdim=True, p=pp),
                # 随机高斯模糊
                k.RandomGaussianBlur(kernel_size=(((3, 3) if strength < 3 else (5, 5)) if strength < 5 else (7, 7)),
                                     sigma=(1.0, 1.0), keepdim=True,
                                     p=pp),
                # 随机裁剪[0.8-1.0]
                self.crop_list[strength - 1],
                # 随机亮度调整
                k.RandomBrightness(brightness=(1.05 + 0.05 * strength, 1.05 + 0.05 * strength), keepdim=True, p=pp),
                # 随机高斯噪声(0.05-0.2)
                k.RandomGaussianNoise(std=(0.0 + 0.05 * (strength if strength < 5 else 4)), keepdim=True, p=pp),
                # 按照原始顺序依次按概率使用图像增强
                random_apply=False,
            )
            x = combine(x)
        return x


class TestDataAugmentationCuda(nn.Module):
    def __init__(self, img_size, device):
        super(TestDataAugmentationCuda, self).__init__()
        self.img_size = img_size
        self.device = device
        self.crop_list = [k.RandomCrop(size=(img_size, img_size), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.95), int(img_size * 0.95)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.9), int(img_size * 0.9)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.85), int(img_size * 0.85)), keepdim=True, p=0.5),
                          k.RandomCrop(size=(int(img_size * 0.8), int(img_size * 0.8)), keepdim=True, p=0.5)]
        # 不同增强的组合
        self.combine = k.AugmentationSequential(
            # 随机JPEG压缩
            k.RandomJPEG(jpeg_quality=(50.0, 90.0), keepdim=True, p=0.5),
            # 随机擦除
            k.RandomErasing(scale=(0.05, 0.2), keepdim=True, p=0.5),
            # 随机旋转
            k.RandomRotation(degrees=(-30, 30), keepdim=True, p=0.5),
            # 随机高斯模糊
            k.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.0, 1.0), keepdim=True, p=0.5),
            # 随机裁剪[0.8-1.0]
            aug_crop(5, crop_list=self.crop_list),
            # 随机亮度调整
            k.RandomBrightness(brightness=(0.7, 1.3), keepdim=True, p=0.5),
            # 随机高斯噪声(0.05-0.2)
            k.AugmentationSequential(
                k.RandomGaussianNoise(std=0.05, keepdim=True, p=0.5),
                k.RandomGaussianNoise(std=0.1, keepdim=True, p=0.5),
                k.RandomGaussianNoise(std=0.15, keepdim=True, p=0.5),
                k.RandomGaussianNoise(std=0.2, keepdim=True, p=0.5),
                random_apply=1
            ),
            # 按照原始顺序依次按概率使用图像增强
            random_apply=False,
        )

    def forward(self, aug_type, strength, x):
        aug = k.AugmentationSequential().to(self.device)
        if aug_type == 'Erasing':
            aug.append(k.RandomErasing(scale=strength, keepdim=True, p=1))
        if aug_type == 'Rotation':
            aug.append(k.RandomRotation(degrees=strength, keepdim=True, p=1))
        if aug_type == 'Blur':
            aug.append(k.RandomGaussianBlur(kernel_size=strength, sigma=(1.0, 1.0), keepdim=True, p=1))
        if aug_type == 'Crop':
            aug.append(k.RandomCrop(size=strength, keepdim=True, p=1))
        if aug_type == 'Brightness':
            aug.append(k.RandomBrightness(brightness=strength, keepdim=True, p=1))
        if aug_type == 'Noise':
            aug.append(k.RandomGaussianNoise(std=strength, keepdim=True, p=1))
        if aug_type == 'JPEG':
            aug.append(k.RandomJPEG(jpeg_quality=strength, keepdim=True, p=1))
        if aug_type == 'Combine':
            # wouaf中的设置为p=1
            pp = 1
            aug = k.AugmentationSequential(
                # 随机JPEG压缩
                k.RandomJPEG(jpeg_quality=(100.0 - strength * 10, 100.0 - strength * 10), keepdim=True, p=pp),
                # 随机擦除
                k.RandomErasing(
                    scale=(0.05 * (strength if strength < 5 else 4), 0.05 * (strength if strength < 5 else 4)),
                    keepdim=True, p=pp),
                # 随机旋转
                k.RandomRotation(degrees=(5 + 5 * strength, 5 + 5 * strength), keepdim=True, p=pp),
                # 随机高斯模糊
                k.RandomGaussianBlur(kernel_size=(((3, 3) if strength < 3 else (5, 5)) if strength < 5 else (7, 7)),
                                     sigma=(1.0, 1.0), keepdim=True,
                                     p=pp),
                # 随机裁剪[0.8-1.0]
                self.crop_list[strength - 1],
                # 随机亮度调整
                k.RandomBrightness(brightness=(1.05 + 0.05 * strength, 1.05 + 0.05 * strength), keepdim=True, p=pp),
                # 随机高斯噪声(0.05-0.2)
                k.RandomGaussianNoise(std=(0.0 + 0.05 * (strength if strength < 5 else 4)), keepdim=True, p=pp),
                # 按照原始顺序依次按概率使用图像增强
                random_apply=False,
            )
        aug = aug.to(self.device)
        x = aug(x)
        return x


class Color:
    # BT601 数字Full range
    def __init__(self):
        pass

    def YUV2RGB(self, image):
        Y = image[:][0]
        U = image[:][1]
        V = image[:][2]
        R = Y + ((360 * (V - 128)) >> 8)
        G = Y - (((88 * (U - 128) + 184 * (V - 128))) >> 8)
        B = Y + ((455 * (U - 128)) >> 8)
        image_rgb = torch.cat([R, G, B], dim=1)
        return image_rgb

    def RGB2YUV(self, image):
        R = image[:][0]
        G = image[:][1]
        B = image[:][2]
        Y = (77 * R + 150 * G + 29 * B) >> 8
        U = ((-44 * R - 87 * G + 131 * B) >> 8) + 128
        V = ((131 * R - 110 * G - 21 * B) >> 8) + 128
        image_yuv = torch.cat([Y, U, V], dim=1)
        return image_yuv

    def YUV2RGBF(self, image):
        # R,G,B (0.0~255.0)
        Y = image[:][0]
        U = image[:][1]
        V = image[:][2]
        R = Y + 1.4075 * (V - 128)
        G = Y - 0.3455 * (U - 128) - 0.7169 * (V - 128)
        B = Y + 1.779 * (U - 128)
        image_rgb = torch.cat([R, G, B], dim=1)
        return image_rgb

    def RGB2YUVF(self, image):
        # Y(0.0~255.0),  U,V(-128.0~128.0)
        R = image[:][0]
        G = image[:][1]
        B = image[:][2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.169 * R - 0.331 * G + 0.5 * B + 128
        V = 0.5 * R - 0.419 * G - 0.081 * B + 128
        image_yuv = torch.cat([Y, U, V], dim=1)
        return image_yuv
