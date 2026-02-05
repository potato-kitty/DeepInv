import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import math
from sklearn.cluster import KMeans
import os
from skimage import morphology
import torch.nn.functional as F
import inspect
from operator import *
import copy
import random
import time
from model import *
from torchvision import transforms

height = 1024 
width = 1024

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def image2latent(model, image):
    device = model.device

    half = model.vae.dtype == torch.float16
    bf = model.vae.dtype == torch.bfloat16

    image = np.array(image)
    image = (torch.from_numpy(image).float() / 127.5) - 1
    if half or bf:
        model.vae.to(dtype=torch.float32)

    image = image.to(device)
    latents = model.vae.encode(image).latent_dist.sample()
    latents = (latents - model.vae.config.shift_factor) * model.vae.config.scaling_factor
    if half:
        model.vae.to(dtype=torch.float16)
        latents = latents.to(dtype=torch.float16)
    elif bf:
        model.vae.to(dtype=torch.bfloat16)
        latents = latents.to(dtype=torch.bfloat16)
    return latents


def mask_reshape(original_mask,res):
    new_mask = np.zeros([res,res])
    att_mask_here = original_mask
    if att_mask_here.shape[0] < new_mask.shape[0]:
        ratio = int(new_mask.shape[0]/att_mask_here.shape[0])
        for i in range(att_mask_here.shape[0]):
            for j in range(att_mask_here.shape[1]):
                new_i = i*ratio
                new_j = j*ratio
                new_mask[new_i:new_i+ratio,new_j:new_j+ratio] = att_mask_here[i,j]
        att_mask_here = new_mask
    elif att_mask_here.shape[0] > new_mask.shape[0]:
        ratio = int(att_mask_here.shape[0]/new_mask.shape[0])
        for i in range(new_mask.shape[0]):
            for j in range(new_mask.shape[1]):
                new_i = i*ratio
                new_j = j*ratio
                new_mask[i,j] = att_mask_here[new_i,new_j]
    
    return new_mask



def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def latent2image_tensor(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255)
    return image


def init_latent(latent, model, height, width, generator, batch_size):

    if latent is None:
        latent = torch.randn(
            (1, model.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.config.in_channels, height // 8, width // 8).to(device)
    return latent, latents

def step_forward(latents,model,prompt_embeds_input,guidance_scale,t):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_input).sample
    noise_pred = noise_pred.sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = model.scheduler.scale_model_input(latents,t)
    latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    return latents, noise_pred

def step_forward_noise(latents,model,prompt_embeds_input,guidance_scale,t):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_input).sample
    noise_pred = noise_pred.sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    return noise_pred

def step_backward(model,latents,prompt_embeds_input,guidance_scale,t):

    model_inputs = torch.cat([latents] * 2)

    noise_pred = model.unet(model_inputs, t, encoder_hidden_states=prompt_embeds_input).sample
    noise_pred = noise_pred.sample

    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

    return noise_pred



@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    guidance_scale=7.0,
    num_round = 1,
    if_train = False,
    order = 0,
    batch_size= 8,
    epochs = 500,
    if_save = True,
    if_shuffle = True,
    device2 = torch.device('cuda:1'),
    tranin_mode = 'real',
    gap_output = 20,
    new_model=False,
    condition = False,
    train_with_add = False,
    itr_idx = 0,
    fr_round = False,
    num_layers = 5,
    use_new_model = False,
    full_train = True,
    lr = 1e-4,
):
    global device
    device = model.device

    if num_round > 1:
        prompt = ['']
        generator = None

    prompt = ['']*batch_size
    negative_prompt = ['']*batch_size
    generator = None
    latent_out,s_net = train_our_model(model,prompt,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,epochs=epochs,if_save = if_save,itr_idx = itr_idx,num_layers = num_layers,full_train = full_train,
                                        guidance_scale = guidance_scale,generator=generator,if_train=if_train,gap_output=gap_output,lr=lr,if_shuffle=if_shuffle,fr_round = fr_round,use_new_model = use_new_model,
                                        device=device,device4=device2,tranin_mode = tranin_mode,new_model=new_model,condition = condition,train_with_add = train_with_add)
    
    return None
