import argparse
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
import torch.nn as nn
import numpy as np
import torch
from torchvision import transforms
import utils
import utils_img
import utils_model
from torchvision.utils import save_image
from src.ldm.models.autoencoder import AutoencoderKL
from src.ldm.models.diffusion.ddpm import LatentDiffusion
from watermark.util import param_util
from watermark.losses import losses
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, StableDiffusionPipeline, \
    UniPCMultistepScheduler, EulerDiscreteScheduler
from watermark.model.watermark_net import WaterMarkGenNet, gen_null_watermark
from tqdm.auto import tqdm
from PIL import Image
import json
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
import pandas as pd
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import webdataset as wds
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('src')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--val_dir", type=str, default='/media/wang004/data/panjiajun/data/datasets/mscoco/val',
        help="Path to the validation data directory")
    group = parser.add_argument_group('Model parameters')
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
    aa("--num_keys", type=int, default=1000, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=1)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def main(params):
    
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

    # 通过归一化将vqgan的输出转为图像
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    # 复制解码器用于微调，将编码器和量化卷积层设置为恒等层
    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.encoder = nn.Identity()
    ldm_decoder.quant_conv = nn.Identity()
    channels, channels_size = param_util.get_param_size_wouaf(ldm_decoder.decoder)
    
    '''
    model_path = 'checkpoints/mscoco_35/'
    status = torch.load(os.path.join(params.output_dir, model_path + "checkpoint_14.pth"))
    ldm_decoder.load_state_dict(status.get('ldm_decoder'))
    wg_net = torch.load(os.path.join(params.output_dir, model_path + "wg_net_14.pth"))
    ke_net = torch.load(os.path.join(params.output_dir, model_path + "ke_net_14.pth"))
    '''
    status = torch.load(os.path.join(params.output_dir, f"checkpoint_000.pth"))
    ldm_decoder.load_state_dict(status.get('ldm_decoder'))
    
    ldm_decoder.encoder = ldm_ae.encoder
    ldm_decoder.quant_conv = ldm_ae.quant_conv
    ldm_decoder.eval()
    ldm_decoder.to(device)
    captions_list = get_caption(type='val')
    # sample_fid_w(ldm_decoder, channels_size, wg_net, params, ke_net, captions_list)
    # status = torch.load(os.path.join(params.output_dir, "checkpoint_005.pth"))
    # ldm_decoder.load_state_dict(status.get('ldm_decoder'))
    sample_fid_stable_signature(params, ldm_decoder, channels_size)
    # captions_list = get_laion_caption()
    # sample_fid_stable_signature(params, ldm_decoder, channels_size)
    # sample_fid(params)

@torch.no_grad()
def sample_fid(params):
    # sd_sample_orig_caption和mscoco512的fid: 19.94463
    real_cache_dir = '/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub/'
    model_id = 'stable-diffusion-2-1-base'
    '''
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", cache_dir=real_cache_dir)
    
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14",
                                              cache_dir="E:\checkpoints\hugging_face_model\huggingface\hub")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                                 cache_dir="E:\checkpoints\hugging_face_model\huggingface\hub")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
                                                subfolder="unet",
                                                cache_dir="E:\checkpoints\hugging_face_model\huggingface\hub")
    '''
    model_path = "/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub/stable-diffusion-2-1-base"
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # 5. move the models to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # 6. set parameters
    prompt = [""]
    height = 512
    width = 512
    # 去噪步数，按照稳定签名设置
    num_inference_steps = 50
    # 设置prompt的权重，越大越接近（默认为7.5）
    guidance_scale = 3.0
    generator = torch.manual_seed(params.seed)
    batch_size = 1
    # 初始编号
    num = 0
    captions_list = get_caption()
    for i in range(int(5000 / batch_size)):
        # 7. get the text_embeddings for the passed prompt
        prompt = captions_list[i]
        # 使用tokenizer编码prompt信息
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        # 8. get the unconditional text embeddings for classifier-free guidance
        # They need to have the same shape as the conditional text_embeddings (batch_size and seq_length)
        max_length = text_input.input_ids.shape[-1]
        # 由于无类别条件输入，因此编码空信息
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        # 9.concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
        # 拼接无类别条件嵌入和文本嵌入
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 10.generate the initial random noise
        # 根据图像尺寸，生成随机潜在输入
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        # 11. initialize the scheduler with our chosen num_inference_steps.
        scheduler.set_timesteps(num_inference_steps)
        # 12. The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here:
        latents = latents * scheduler.init_noise_sigma
        # 13. write the denoising loop
        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # noise_pred = unet(latent_model_input, t).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 14. use the vae to decode the generated latents back into the image
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = vae.decode(latents).sample

        for j in range(batch_size):
            # 15. convert the image to PIL so we can display or save it
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            pil_images[j].save(f"/media/wang004/data/panjiajun/data/image_samples/sd2-1-base_sample_orig_DPMSolver/{num}.png")
            num += 1

@torch.no_grad()
def sample_fid_laion(params):
    real_cache_dir = '/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub'
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                        cache_dir=real_cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                              cache_dir=real_cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
        cache_dir=real_cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet",
        cache_dir=real_cache_dir
    )
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    prompt = [""]
    height = 512
    width = 512
    num_inference_steps = 50
    guidance_scale = 3.0
    generator = torch.manual_seed(params.seed)
    batch_size = 1
    num = 0
    captions_list = get_laion_caption()
    for i in range(int(5000 / batch_size)):
        prompt = captions_list[i]
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.init_noise_sigma
        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = vae.decode(latents).sample

        for j in range(batch_size):
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            pil_images[j].save(f"/media/wang004/data/panjiajun/data/image_samples/sd_sample_orig_laion/{num}.png")
            num += 1


@torch.no_grad()
def sample_fid_stable_signature(params, ldm_decoder, channels_size):
    # 稳定签名设置:生成5k图像，与验证集的5k图像计算FID，使用验证集图像的首个描述信息，指导系数为3.0
    '''
    real_cache_dir = '/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub'
    vae = ldm_decoder
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                              cache_dir=real_cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
        cache_dir=real_cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet",
        cache_dir=real_cache_dir
    )
    '''
    model_path = "/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub/stable-diffusion-2-1-base"
    vae = ldm_decoder
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')

    # 4. load the K-LMS scheduler with some fitting parameters
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # 5. move the models to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # 6. set parameters
    prompt = [""]
    height = 512
    width = 512
    # 去噪步数，按照稳定签名设置
    num_inference_steps = 50
    # 设置prompt的权重，越大越接近（默认为7.5）
    guidance_scale = 3.0
    generator = torch.manual_seed(params.seed)
    batch_size = 1
    num = 0
    captions_list = get_caption()
    for i in range(int(5000 / batch_size)):
        # 7. get the text_embeddings for the passed prompt
        prompt = captions_list[i]
        # 7. get the text_embeddings for the passed prompt
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        # 8. get the unconditional text embeddings for classifier-free guidance
        # They need to have the same shape as the conditional text_embeddings (batch_size and seq_length)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        # 9.concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 10.generate the initial random noise
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        # 11. initialize the scheduler with our chosen num_inference_steps.
        scheduler.set_timesteps(num_inference_steps)
        # 12. The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here:
        latents = latents * scheduler.init_noise_sigma
        # 13. write the denoising loop
        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # noise_pred = unet(latent_model_input, t).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 14. use the vae to decode the generated latents back into the image
        latents = 1 / 0.18215 * latents
        w = gen_null_watermark(channels_size, device)
        with torch.no_grad():
            image = vae.decode(latents, w)
            # decoded = msg_decoder(vqgan_to_imnet(image))
            # bitacc = losses.get_bit_acc(decoded, key)
            # print('bitacc {:.4f}'.format(bitacc))
        for j in range(batch_size):
            # 15. convert the image to PIL so we can display or save it
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            pil_images[j].save(f"/media/wang004/data/panjiajun/data/image_samples/sd2-1-base_sample_stable_signature_coco/{num}.png")
            num += 1


@torch.no_grad()
def sample_fid_stable_signature_laion(params, ldm_decoder, channels_size, captions_list):
    # 稳定签名设置:生成5k图像，与验证集的5k图像计算FID，使用验证集图像的首个描述信息，指导系数为3.0
    real_cache_dir = '/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub'
    vae = ldm_decoder
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                              cache_dir=real_cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
        cache_dir=real_cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet",
        cache_dir=real_cache_dir
    )
    # 4. load the K-LMS scheduler with some fitting parameters
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # 5. move the models to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # 6. set parameters
    prompt = [""]
    height = 512
    width = 512
    # 去噪步数，按照稳定签名设置
    num_inference_steps = 50
    # 设置prompt的权重，越大越接近（默认为7.5）
    guidance_scale = 3.0
    generator = torch.manual_seed(params.seed)
    batch_size = 1
    num = 0
    for i in range(int(1000 / batch_size)):
        # 7. get the text_embeddings for the passed prompt
        prompt = captions_list[i]
        # 7. get the text_embeddings for the passed prompt
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        # 8. get the unconditional text embeddings for classifier-free guidance
        # They need to have the same shape as the conditional text_embeddings (batch_size and seq_length)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        # 9.concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 10.generate the initial random noise
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        # 11. initialize the scheduler with our chosen num_inference_steps.
        scheduler.set_timesteps(num_inference_steps)
        # 12. The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here:
        latents = latents * scheduler.init_noise_sigma
        # 13. write the denoising loop
        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # noise_pred = unet(latent_model_input, t).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 14. use the vae to decode the generated latents back into the image
        latents = 1 / 0.18215 * latents
        w = gen_null_watermark(channels_size, device)
        with torch.no_grad():
            image = vae.decode(latents, w)
            # decoded = msg_decoder(vqgan_to_imnet(image))
            # bitacc = losses.get_bit_acc(decoded, key)
            # print('bitacc {:.4f}'.format(bitacc))
        for j in range(batch_size):
            # 15. convert the image to PIL so we can display or save it
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            pil_images[j].save(f"/media/wang004/data/panjiajun/data/image_samples/sd_sample_stable_signature_laion/{num}.png")
            num += 1



@torch.no_grad()
def sample_fid_w(ldm_decoder, channels_size, wg_net, params, ke_net, captions_list):
    '''
    # ke_net = torch.load(os.path.join(params.output_dir, "ke_net.pth"))
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = ldm_decoder
    real_cache_dir = '/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub'
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer",
                                              cache_dir=real_cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder",
        cache_dir=real_cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet",
        cache_dir=real_cache_dir
    )
    # 4. load the K-LMS scheduler with some fitting parameters / wouaf使用euler调度器
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

    # scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler",
    #                                                     cache_dir="E:\checkpoints\hugging_face_model\huggingface\hub")
    '''
    model_path = "/media/wang004/data/panjiajun/checkpoints/hugging_face_model/huggingface/hub/stable-diffusion-2-1-base"
    
    vae = ldm_decoder
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet",)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder='scheduler')

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # 5. move the models to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # 6. set parameters
    prompt = [""]
    height = 512
    width = 512
    # 去噪步数，按照稳定签名设置
    num_inference_steps = 50
    # 设置prompt的权重，越大越接近（默认为7.5）
    guidance_scale = 3.0
    generator = torch.manual_seed(params.seed)
    batch_size = 1
    sum = 5000
    key_list = get_key_list(key_num=sum)
    w = []
    for i in range(int(sum / batch_size)):
        key = key_list[i]
        wa, m = wg_net(key)
        w_temp = []
        w_temp.append(wa)
        w_temp.append(0)
        w_temp.append(0)
        w.append(w_temp)
    wg_net = nn.Identity()
    num = 0
    for i in range(0, int(sum / batch_size)):
        # 7. get the text_embeddings for the passed prompt
        prompt = captions_list[i]
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        # 8. get the unconditional text embeddings for classifier-free guidance
        # They need to have the same shape as the conditional text_embeddings (batch_size and seq_length)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        # 9.concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 10.generate the initial random noise
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        # 11. initialize the scheduler with our chosen num_inference_steps.
        scheduler.set_timesteps(num_inference_steps)
        # 12. The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here:
        latents = latents * scheduler.init_noise_sigma
        # 13. write the denoising loop
        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 14. use the vae to decode the generated latents back into the image
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = vae.decode(latents, w[i])

        for j in range(batch_size):
            # extract_key = ke_net(image)
            # bit_acc = losses.get_bit_acc(extract_key, key_list[i])
            # 15. convert the image to PIL so we can display or save it
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            # print("bit_acc:{:.4f}".format(bit_acc))
            pil_images[j].save(f"/media/wang004/data/panjiajun/data/image_samples/sd2-1-base_sample_wouaf_coco/{num}.png")
            num += 1


def change_mscoco_size(params):
    transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
    ])
    val_loader = utils.get_dataloader(params.val_dir, transform, 1, num_imgs=None, shuffle=True, num_workers=4,
                                      collate_fn=None)
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    for ii, imgs in enumerate(metric_logger.log_every(val_loader, params.log_freq, header)):
        save_image(torch.clamp(imgs, 0, 1), os.path.join(f'/media/wang004/data/panjiajun/data/image_samples/mscoco_512/{ii}.png'), nrow=1)


def get_test_webdataset_loader():
    # 定义预处理流程，移除wds.decode("pil")
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    dataset = (
        wds.WebDataset("/media/wang004/data/panjiajun/data/datasets/laion2B-en-aesthetic/{01413..01568}.tar"
        ,handler=wds.warn_and_continue)
        .decode("pil")
        .to_tuple("jpg")
        .map_tuple(preprocess)
        .batched(1)
    )
    # 创建DataLoader
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4)
    return dataloader


def get_test_json_webdataset_loader():

    dataset = (
        wds.WebDataset("/media/wang004/data/panjiajun/data/datasets/laion2B-en-aesthetic/{01413..01568}.tar" 
        ,handler=wds.warn_and_continue)
        .to_tuple("txt")
        .batched(1)
    )
    # 创建DataLoader
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4)
    return dataloader


def change_laion_size():
    val_loader = get_test_webdataset_loader()
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ii = 0
    for imgs in val_loader:
        save_image(torch.clamp(imgs[0], 0, 1), os.path.join(f'/media/wang004/data/panjiajun/data/image_samples/laion_512/{ii}.png'), nrow=1)
        ii += 1
        if ii > 1000:
            break


def get_laion_caption():
    val_loader = get_test_json_webdataset_loader()
    ii = 0
    captions_list = []
    for text in val_loader:
        captions_list.append(text[0][0].decode('utf-8'))
        ii += 1
        if ii > 999:
            break
    return captions_list
    

def get_caption(type='val'):
    root = f'/media/wang004/data/panjiajun/data/datasets/mscoco/annotations/captions_{type}2017.json'
    with open(root, 'r') as file:
        annotations = json.load(file)
    # annotations包含图像的文字描述信息，image_id为图像的id，每个图像可能有多个描述信息
    captions = annotations['annotations']
    s = set()
    captions_list = []
    for i in range(len(captions)):
        if captions[i]['image_id'] in s:
            continue
        s.add(captions[i]['image_id'])
        captions_list.append(captions[i]["caption"])
    return captions_list


# 获取用于生成指纹图像的指纹集
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


def calculate_clip_score(images, prompts):
    # import pdb;pdb.set_trace()
    # images_int = (np.asarray(images[0]) * 255).astype("uint8")
    clip_score_fn = partial(clip_score,
                            model_name_or_path="E:\checkpoints\hugging_face_model\huggingface\hub\models--openai--clip-vit-large-patch14\check")
    images_int = (np.asarray(images) * 255).astype("uint8")
    images_int = np.expand_dims(images_int, axis=0)
    # images_int = images
    clip_scores = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_scores), 4)


def get_clip_score():
    captions_list = get_val_caption()
    cs = 0
    images_folder_path = 'E:\image_samples\sd_sample_orig_caption/'
    for i in range(len(captions_list)):
        image_path = images_folder_path + f'{i}.jpg'
        image = Image.open(image_path)
        cs += calculate_clip_score(image, captions_list[i])
    print('avg_clip_score: ', cs / len(captions_list))


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    main(params)
