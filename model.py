import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
import inspect
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
import copy
import os
import random
from PIL import Image
import numpy as np
import utils
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection, AutoProcessor, CLIPVisionModel
import cv2
from IPython.display import display
import test
from torchvision import transforms
from torchvision.utils import save_image
from ssim import ms_ssim
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
import torchviz
from torch.optim import AdamW
import json
import time
#from torch.nn.utils import freeze_params

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
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
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def SD3(
    model,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    in_latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    invert_tst = False,
    our_method = False,
    device = torch.device('cuda:0'),
    if_grad = False,
    noised_latent_list = None,
    alpha = 0.5,
    idx_in = 0,
    if_train = True,
    real_img_idx = None,
    add_weight = 0,
    out_data_path = None,
):
    if noised_latent_list is not None:
        noised_latent_list = list(reversed(noised_latent_list))
    with torch.enable_grad() if if_grad else torch.no_grad():
        height = height or model.default_sample_size * model.vae_scale_factor
        width = width or model.default_sample_size * model.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        model.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        model._guidance_scale = guidance_scale
        model._clip_skip = clip_skip
        model._joint_attention_kwargs = joint_attention_kwargs
        model._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = model.device
        #model = model.to(device)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=model.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=model.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        #print('prompt_embeds: ',prompt_embeds[0,-1,:])

        if model.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(model.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)
        model._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = model.transformer.config.in_channels
        latents = model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            in_latents,
        )

        # 6. Denoising loop

        pred_noise_list = []
        encode_hidd_list = []
        save_latents_list = []
        idx = 0
        num_add = len(noised_latent_list)
        for i, t in enumerate(timesteps):
            if model.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if model.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = model.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=model.joint_attention_kwargs,
                return_dict=False,
            )
            noise_pred = noise_pred[0]

            if our_method:
                #pred_latent_in_list.append(latent_model_input.detach().cpu())
                pred_noise_list.append(noise_pred.detach().cpu())

            # perform guidance
            if model.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_dtype = latents.dtype
            
            last_latents = latents.clone()
            latents,dt,sigma_hat,sigma = model.scheduler.step(noise_pred, t, latents, return_dict=False,recover=True)

            if noised_latent_list is not None:
                if i < len(timesteps)*add_weight:
                    if num_add > 1:
                        latents = alpha*latents + (1-alpha)*noised_latent_list[idx].to(device)
                    else:
                        noise_pred = alpha*noise_pred + (1-alpha)*noised_latent_list[0].to(device)
                    add_noise = (((latents - last_latents)/dt)*sigma_hat)/sigma
                else:
                    add_noise = noise_pred
                
                ratio = int(len(timesteps)/num_add)
                if (i+1)%ratio == 0:
                    idx += 1

                if not if_train:
                    save_latents_list.append(add_noise)

            

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(model, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )

        if output_type == "latent":
            image = latents
        else:
            show_latents = (latents / model.vae.config.scaling_factor) + model.vae.config.shift_factor
            latents = latents.detach()
            images = []
            for idx in range(len(show_latents)):
                show_latents_tmp = show_latents[idx].unsqueeze(0)
                image = model.vae.decode(show_latents_tmp, return_dict=False)[0]
                image = model.image_processor.postprocess(image, output_type=output_type)
                images.append(image[0])

        # Offload all models
        model.maybe_free_model_hooks()

        if not if_train:
            while True:
                if out_data_path is None:
                    save_str = './DeepInv/real_img_denoise_latents_itr_2/' + str(idx_in) + '.pt'
                    save_idx_path = './DeepInv/real_img_denoise_latents_itr_2/' + str(idx_in) + '.txt'
                else:
                    save_str = out_data_path + str(idx_in) + '.pt'
                    save_idx_path = out_data_path + str(idx_in) + '.txt'
                idx_in += 1
                if not os.path.exists(save_str):
                    break

            torch.save(save_latents_list,save_str)
            if real_img_idx is not None:
                with open(save_idx_path, 'w') as f:
                    json.dump(real_img_idx, f)




        if our_method:
            return images, latents,torch.stack(pred_noise_list)
        elif invert_tst:
            return images, latents, encode_hidd_list, pred_noise_list
        else:
            return images

class DeepInvModel(nn.Module):
    def __init__(
                self,
                num_head = 24,
                num_head_dim = 64,
                num_layers = 4,
                pooled_projection_dim = 2048,
                patch_size = 2,
                out_channels = 16,
                height = 1024,
                width = 1024,
                inchannels = 64,
                latent_size = 128,
                pos_embed_max_size: int = 192,
                joint_attention_dim: int = 4096,
                ):
        super().__init__()
        
        self.num_head = num_head
        self.num_head_dim = num_head_dim
        self.inner_dim = self.num_head*self.num_head_dim
        self.num_layers = num_layers
        self.pooled_projection_dim = pooled_projection_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.in_channels = inchannels
        self.joint_attention_dim = joint_attention_dim
        self.latent_size = latent_size

        self.blocks = nn.ModuleList([])
        self.blocks_img = nn.ModuleList([])
        #self.block_res = nn.ModuleList([])
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.pooled_projection_dim
        )
        self.time_text_embed2 = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.pooled_projection_dim
        )
        self.pos_embed1 = PatchEmbed(
            height=self.height,
            width=self.width,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        
        self.pos_embed2 = PatchEmbed(
            height=self.height,
            width=self.width,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        
        self.context_embedder = nn.Linear(self.joint_attention_dim, self.inner_dim)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.res1 = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1)

        self.norm_out_img = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        #self.proj_out_img = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.pool_conv = nn.Conv2d(out_channels,1,kernel_size=1,stride=1)
        self.pool_full = nn.Linear(latent_size*latent_size, pooled_projection_dim, bias=True)
        
        #self.res2 = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        for i in range(self.num_layers):
            self.blocks.append(JointTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.num_head,
                        attention_head_dim=self.inner_dim,
                        context_pre_only=i == self.num_layers - 1 and not(self.num_layers == 1),
                        #context_pre_only=False,
                    ))
            self.blocks_img.append(JointTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.num_head,
                        attention_head_dim=self.inner_dim,
                        context_pre_only=i == self.num_layers - 1 and not(self.num_layers == 1),
                        #context_pre_only=False,
                    ))
            
        self.combined_block = JointTransformerBlock(
            dim=self.inner_dim,
            num_attention_heads=self.num_head,
            attention_head_dim=self.inner_dim,
            context_pre_only=True,
        )
            #self.block_res.append()
    
    def forward(self, timestep, hidden_states, pooled_projections: torch.FloatTensor = None,encoder_hidden_states: torch.FloatTensor = None,
                en_hds_img = None,pool_hds_img = None, 
                if_traning = False,num_itr = 1, *args: Any, **kwds: Any):


        out_list = []

        height, width = hidden_states.shape[-2:]

        in_noise = hidden_states.clone()

        batch_size = pooled_projections.shape[0]
        pool_hds_img = self.pool_conv(pool_hds_img)
        pool_hds_img = self.pool_full(pool_hds_img.view(batch_size,-1))
        
        hidden_states = self.pos_embed1(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        en_hds_img = self.pos_embed2(en_hds_img)
        hidden_states_img = hidden_states.clone()

        temb = self.time_text_embed(timestep, pooled_projections)
        temb_img = self.time_text_embed2(timestep, pool_hds_img)

        for block,block_img in zip(self.blocks,self.blocks_img):
            if if_traning:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                for idx in range(num_itr):
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    encoder_hidden_states, hidden_states_out = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = hidden_states + hidden_states_out

                    en_hds_img, hidden_states_img_out = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block_img),
                        hidden_states_img,
                        en_hds_img,
                        temb_img,
                        **ckpt_kwargs,
                    )

                    hidden_states_img = hidden_states_img + hidden_states_img_out
            else:
                for idx in range(num_itr):
                    encoder_hidden_states, hidden_states_out = block(
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                    )
                    hidden_states = hidden_states + hidden_states_out

                    en_hds_img, hidden_states_img_out = block_img(
                        hidden_states_img,
                        en_hds_img,
                        temb_img,
                    )
                    hidden_states_img = hidden_states_img + hidden_states_img_out
            
        hidden_states = self.norm_out(hidden_states, temb)

        hidden_states_img = self.norm_out_img(hidden_states_img, temb_img)

        hidden_states_img, hidden_states_com  = self.combined_block(
            hidden_states,
            hidden_states_img,
            temb_img,
        )

        hidden_states = self.proj_out(hidden_states + hidden_states_com)

        # unpatchify
        patch_size = self.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        output = in_noise + output
        
        return output,out_list
    

def prompt_encoder(
    model,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    prompt_embeds = None,
    negative_prompt_embeds = None,
    pooled_prompt_embeds = None,
    negative_pooled_prompt_embeds = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    max_sequence_length: int = 256,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    height = None,
    width = None,
    our_encode = False,
    projected_emb = None,
    pool_img_emb = None,
    ):

    device = model.device
    height = height or model.default_sample_size * model.vae_scale_factor
    width = width or model.default_sample_size * model.vae_scale_factor
    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    model._clip_skip = clip_skip
    model._joint_attention_kwargs = joint_attention_kwargs
    model._interrupt = False

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = model.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=model.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=model.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    '''
    if model.do_classifier_free_guidance:
        prompt_embeds_ours = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds_ours = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        prompt_embeds_ours = prompt_embeds
        pooled_prompt_embeds_ours = pooled_prompt_embeds
    '''
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    #,prompt_embeds_ours,pooled_prompt_embeds_ours

def train_our_model(
    model,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    lr = 1e-4,
    full_train = True,
    weight_decay = 1e-4,
    if_train = True,
    epochs = 1000,
    gap_output = 100,
    if_save = False,
    new_model = True,
    use_new_model = False,
    if_shuffle = False,
    device4 = torch.device('cuda:1'),
    device = torch.device('cuda:0'),
    tranin_mode = 'real',
    condition = False,
    train_with_add = False,
    itr_idx = 0,
    fr_round = False,
    num_layers = 9,
    if_COCO = True,
):

    if_COCO = False

    gt_latents_idx_list = [1,0.8,0.6,0.5]
    if_tst = True
    if itr_idx == -1:
        add_weight = 0.1
        itr_idx = 4
        if_tst = True
    else:
        if itr_idx + 1 < len(gt_latents_idx_list):
            add_weight = gt_latents_idx_list[itr_idx + 1]
        else:
            add_weight = gt_latents_idx_list[-1]        

    if fr_round:
        gt_latents_idx = 1
    else:
        gt_latents_idx = 1 - gt_latents_idx_list[itr_idx]
    
    add_weight = 0.5
    gt_latents_idx = 1 - 0.5

    height = height or model.default_sample_size * model.vae_scale_factor
    width = width or model.default_sample_size * model.vae_scale_factor

    print('model with ' + str(num_layers) + ' layers')

    print('Iteration ' + str(itr_idx) + ', ' + str(num_inference_steps) + ' steps')
    print('All using latents' if fr_round and train_with_add else 'With mix GT')

    out_path = './DeepInv/final_version/itr_' + str(itr_idx) + '/'
    out_path_last = './DeepInv/final_version/itr_' + str(itr_idx-1) + '/'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    last_layers_num = 5

    in_data_path = './DeepInv/final_version/itr_' + str(itr_idx) + '/out_data_'+str(last_layers_num)+'/'

    if if_tst:
        out_data_path = out_path + 'out_data_'+str(num_layers)+'_tst/'
    else:
        out_data_path = out_path + 'out_data_'+str(num_layers)+'/'
    if not os.path.exists(out_data_path):
        os.mkdir(out_data_path)

    out_img_path = out_path + str(num_layers) + '_layers_with_' + str(num_inference_steps) + '_steps/'
    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)
    
    if if_tst:
        out_tst_path = './DeepInv/final_version/tst_' + str(num_layers) + '_layers'+'_weight_'+str(add_weight)+'/'
        if not os.path.exists(out_tst_path):
            os.mkdir(out_tst_path)
    else:
        out_tst_path = out_path + 'tst_' + str(num_layers) + '_layers/'
        if not os.path.exists(out_tst_path):
            os.mkdir(out_tst_path)
    
    save_model_path = out_path + 'save_model/'
    save_model_path_last = out_path_last + 'save_model/'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    '''
    print('height: ',height)
    print('width: ',width)
    '''
    device = model.device
    batch_size = len(prompt)
    print('batch size is',batch_size)
    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    #guidance_scale = 1
    model._guidance_scale = guidance_scale
    model._clip_skip = clip_skip
    model._joint_attention_kwargs = joint_attention_kwargs
    model._interrupt = False
    print('CFG: ',model.do_classifier_free_guidance)

    # 2. Define call parameters
    
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = model.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=model.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=model.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    #prompt_embeds_ours_vis = prompt_embeds.clone()
    #pooled_prompt_embeds_ours_vis = pooled_prompt_embeds.clone()
    
    if model.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(model.scheduler, num_inference_steps, device, timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)
    model._num_timesteps = len(timesteps)

    # 5. Prepare latent variables
    num_channels_latents = model.transformer.config.in_channels

    clip_model = CLIPVisionModelWithProjection.from_pretrained("./pretrained_models/clip-vit",ignore_mismatched_sizes=True) #WithProjection
    clip_processor = AutoProcessor.from_pretrained("./pretrained_models/clip-vit")
    
    model_path = save_model_path + 's_net_with_'+ str(num_layers) +'layers_itr_' + str(itr_idx) + '.pth'
    print('load model from '+model_path)
    model_path_last = save_model_path_last + 's_net_with_'+ str(num_layers) +'layers_itr_' + str(itr_idx-1) + '.pth'

    if if_train:
        if new_model:
            train_snet = True
            train_encoder = False
            if train_snet:
                if fr_round and num_inference_steps == 1 or use_new_model:
                    if itr_idx > 0 and use_new_model:
                        print('Start training new model!')
                        num_add_layers = int(num_layers - last_layers_num)
                        if not full_train:
                            s_net_7 = DeepInvModel(inchannels=num_channels_latents,num_layers=num_add_layers+1).bfloat16()
                            for param in s_net_7.parameters():
                                param.requires_grad = True
                            s_net = torch.load('./DeepInv/final_version/itr_4/save_model/s_net_with_'+str(last_layers_num)+'layers_itr_4.pth')
                            print('s_net blocks:',len(s_net.blocks))
                            print('s_net blocks_img:',len(s_net.blocks_img))
                            for param in s_net.parameters():
                                param.requires_grad = False
                            for xdi in range(num_add_layers):
                                s_net.blocks.insert(int(xdi*2+1),s_net_7.blocks[xdi].to(device4))
                                #s_net.blocks_img.insert(int(xdi*2+1),s_net_7.blocks_img[xdi].to(device4))
                            print('s_net blocks:',len(s_net.blocks))
                            print('s_net blocks_img:',len(s_net.blocks_img))
                            print('####################### Only add additional blocks! #######################')
                            '''
                            for param in s_net.parameters():
                                print(param,':',param.requires_grad)
                            '''
                        else:
                            print('Train with model from last round!')
                            s_net = torch.load(model_path,map_location=device4)
                            lr = lr/10
                            for param in s_net.parameters():
                                param.requires_grad = True
                            print('####################### !Full Fine-tune! #######################')
                    elif itr_idx == 0 and use_new_model:
                        s_net = DeepInvModel(inchannels=num_channels_latents,num_layers=num_layers).bfloat16()
                    else:
                        print('Train with model from last round!')
                        s_net = torch.load(model_path,map_location=device4)
                        for param in s_net.parameters():
                            param.requires_grad = True
                        if full_train:
                            #lr = lr/100 # lr/50
                            print('####################### Full Fine-tune #######################')
                    #s_net = DeepInvModel(inchannels=num_channels_latents,num_layers=num_layers).bfloat16().to(device4)
                else:
                    print('Train with saved model!')
                    s_net = torch.load(model_path,map_location=device4)
            else:
                s_net = torch.load('./DeepInv/save_model/s_net_'+ str(num_layers) +'_real_img_itr_2.pth',map_location=device4)
                print('load snet!')
        else:
            print('Keep training on previous model!')
            train_snet = True
            train_encoder = False
            #s_net = torch.load('./DeepInv/save_model/s_net_'+ str(num_layers) +'_real_img_itr_1_mix.pth',map_location=device4)
            s_net = torch.load('./DeepInv/save_model/s_net_'+ str(num_layers) +'_real_img_itr_2_mix.pth',map_location=device4)
            print('model with '+len(s_net.blocks)+' layers loaded!')
            encoder_emb = torch.load('./DeepInv/encoder_save/encoder_emb_new.pth',map_location=device4)
            #encoder_emb = torch.load('./DeepInv/save_model/encoder_emb_'+ str(num_layers) +'_real_img.pth',map_location=device4)
        #if condition:
    else:
        train_snet = False
        train_encoder = False
        print('Testing on previous model!')
        s_net = torch.load(model_path,map_location=device4)
        encoder_emb = torch.load('./DeepInv/encoder_save/encoder_emb_new.pth',map_location=device4)

    loss_function_l2 = nn.MSELoss()
    loss_function_l1 = nn.L1Loss()
    if train_snet:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, s_net.parameters()), lr=lr, weight_decay=weight_decay,eps=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6, verbose=False)
    if train_encoder:
        optimizer_emb = torch.optim.AdamW(encoder_emb.parameters(), lr=lr, weight_decay=weight_decay,eps=1e-4)
        scheduler_emb = torch.optim.lr_scheduler.StepLR(optimizer_emb, step_size=30, gamma=0.6, verbose=False)


    print('shuffle') if if_shuffle else print('no shuffle')
    print('train with condition') if condition else print('no condition')
    print('train with real images') if tranin_mode == 'real' else print('train with generated images')
    print('train encoder') if train_encoder else print('no encoder trainning')
    print('train snet') if train_snet else print('no snet trainning')
    #print('train with denoised latents') if train_with_add else print('first round of traning')
    


    num_fixed_itr = 3
 
    if train_encoder and not train_snet:
        num_real_itr = 1  
    else:
        num_real_itr = 3
    
    if not if_train:
        num_real_itr = 1

    in_path = './DeepInv/tst_prompt.txt'

    total_epoch_loss = 0
    total_epoch_reg_loss = 0

    inject_steps = 0.05 #0.05
    inject_len = 0.35 #0.35
    #gt_latents_idx = 0.6

    if if_train:
        img_path = "./COCO/out_img/"
    else:
        img_path = "./COCO/tst_img/"

    if if_tst and not if_COCO:
        num_real_img = 4
    else:
        num_real_img = len(os.listdir(img_path))
    if epochs > num_real_img:
        epochs = num_real_img
    real_img_idx_list = list(range(num_real_img))

    #if train_with_add:
    rand_num_list = [x for x in range(epochs)]
    random.shuffle(rand_num_list)
    
    restar_epoch = 0
    if if_train and fr_round:
        restar_epoch = int(len(os.listdir(out_img_path))/2)*5
        print('restart from itr ' + str(itr_idx) + ' num_inf_steps ' + str(num_inference_steps) + ' epochs ' + str(restar_epoch))

    latent_as_condition = None

    skip_list = []

    with model.progress_bar(total=epochs) as progress_bar:
        for epoch in range(epochs):
            if if_tst:
                start_time = time.perf_counter()
            if epoch < restar_epoch or (epoch in skip_list and if_train):
                progress_bar.update()
                continue

            if if_train:
                gt_latents = torch.load(in_data_path + str(rand_num_list[epoch])+'.pt')
                gt_latents = list(reversed(gt_latents))
            model.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
            total_t_loss = 0
            total_t_reg_loss = 0
            if if_train and tranin_mode == 'gen':
                rand_num = random.randint(0, 9)
                if rand_num < 2:
                    type_str = 'with prompts'
                    with open(in_path,"r") as f:
                        prompts = f.readlines()
                        prompt_c = random.sample(prompts, batch_size)
                    for p_idx, p in enumerate(prompt_c):
                        prompt_c[p_idx] = p.strip('\n')
                else:
                    type_str = 'empty prompts'
                    prompt_c = ['']*batch_size
                generator = torch.Generator().manual_seed(random.randint(1,100000))
                real_img,latent_out,pred_noise_list = SD3(model,prompt_c,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,
                                                        guidance_scale = guidance_scale,generator=generator,our_method=True)

                model.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
                shuffled_idx_list = reversed(list(range(pred_noise_list.shape[0])))
                latent_as_condition = latent_out
            elif tranin_mode == 'real':
                if if_train:
                    with open(in_data_path + str(rand_num_list[epoch]) + '.txt', 'r') as f:
                        real_img_idx = json.load(f)
                elif not if_tst:
                    real_img_idx = random.sample(real_img_idx_list, batch_size)
                elif batch_size > 1 and if_tst:
                    real_img_idx = real_img_idx_list[epoch*batch_size:(epoch+1)*batch_size]
                else:
                    real_img_idx = [real_img_idx_list[epoch]]

                real_img = []
                img_emb = []
                pool_img_emb = []
                for ix,img_idx in enumerate(real_img_idx):

                    if not if_train and if_COCO:
                        img_idx = img_idx+2000
                        real_img_in = cv2.imread(img_path + str(img_idx) + ".jpg")
                    else:
                        real_img_in = cv2.imread('./UltraEdit/images/example_images/' + str(img_idx+1) + "-input.png")
                        

                    real_img_in = cv2.cvtColor(real_img_in, cv2.COLOR_BGR2RGB)
                    real_img_in = cv2.resize(real_img_in,(1024,1024))

                    inputs_img = clip_processor(images=real_img_in, return_tensors="pt", padding=True)
                    outputs_img = clip_model(**inputs_img,if_trans_all=True) #.image_embeds
                    img_emb.append(torch.tensor(outputs_img.last_hidden_state))
                    pool_img_emb.append(torch.tensor(outputs_img.image_embeds))

                    if ix == 0:
                        out_img = Image.fromarray(real_img_in)
                    real_img.append(torch.tensor(real_img_in).permute(2,0,1))

                img_emb = torch.stack(img_emb).bfloat16().to(device4).squeeze()
                pool_img_emb = torch.stack(pool_img_emb).bfloat16().to(device4).squeeze()
                #projected_emb = encoder_emb(img_emb)
                #compare_emb = projected_emb[:,0,:]
                
                real_imgs_in = torch.stack(real_img)
                latent_as_condition = utils.image2latent(model,real_imgs_in)
                or_latent = copy.deepcopy(latent_as_condition).to(device4)
                or_latent_db = torch.cat([or_latent] * 2) if model.do_classifier_free_guidance else or_latent

                (prompt_embeds_or, 
                negative_prompt_embeds, 
                pooled_prompt_embeds_or, 
                negative_pooled_prompt_embeds,) = prompt_encoder(model,prompt=prompt)
                #print('CFG: ',model.do_classifier_free_guidance)
                with torch.enable_grad():
                    if model.do_classifier_free_guidance:
                        prompt_embeds_ours = torch.cat([negative_prompt_embeds, prompt_embeds_or], dim=0)
                        pooled_prompt_embeds_ours = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds_or], dim=0)
                    else:
                        prompt_embeds_ours = prompt_embeds_or
                        pooled_prompt_embeds_ours = pooled_prompt_embeds_or

                
                #opt_emb = AdamW([prompt_embeds_or])
                #opt_emb_pool = AdamW([pooled_prompt_embeds_or])
                #opt_emb_neg = AdamW([negative_prompt_embeds])
                #opt_emb_pool_neg = AdamW([negative_pooled_prompt_embeds])            

                prompt_embeds_ours_vis = prompt_embeds_or
                pooled_prompt_embeds_ours_vis = pooled_prompt_embeds_or
                    
            elif not if_train and tranin_mode == 'gen':
                gap_output = 1
                batch_size = 1
                real_imgs_in = cv2.imread("./COCO/out_img/" + str(epoch) + ".jpg")
                real_imgs_in = cv2.cvtColor(real_imgs_in, cv2.COLOR_BGR2RGB)
                real_imgs_in = cv2.resize(real_imgs_in,(1024,1024))
            else:
                real_img_in = cv2.imread("./COCO/out_img/" + str(epoch) + ".jpg")
                real_img_in = cv2.cvtColor(real_img_in, cv2.COLOR_BGR2RGB)
                real_img_in = cv2.resize(real_img_in,(1024,1024))
                real_img_show = copy.deepcopy(real_img_in)

                gen_img_out = gen_img_out[0]

                to_tensor = transforms.ToTensor()
                real_img_in = to_tensor(real_img_in).unsqueeze(0).to(dtype=torch.float32)

                latent_as_condition = utils.image2latent(model,real_img_in)

                or_img_latent = utils.image2latent(model,torch.tensor(real_img_show).unsqueeze(0).permute(0,3,1,2))
                latent_as_condition = 0.8*latent_as_condition + 0.2*or_img_latent
                gt_latent = copy.deepcopy(latent_as_condition)
                real_img_in = Image.fromarray(real_img_show)

            if if_shuffle:
                random.shuffle(shuffled_idx_list)

            num_rand_noise = 0
            minors = 0

            itr_scheduler = copy.deepcopy(model.scheduler)

            if if_train:
                if train_snet:
                    s_net.train()
                if train_encoder:
                    encoder_emb.train()
            else:
                s_net.eval()
            
            noised_latent_list = []

            for i, t in enumerate(reversed(timesteps)):

                prompt_embeds_ours_cp = prompt_embeds_ours.clone()
                pooled_prompt_embeds_ours_cp = pooled_prompt_embeds_ours.clone()
                with torch.no_grad():
                    start_latent = copy.deepcopy(latent_as_condition).to(device)
                    latents_input = torch.cat([start_latent] * 2) if model.do_classifier_free_guidance else start_latent
                    timestep = t.expand(latents_input.shape[0])
                    timestep = timestep.to(device)
                    latents_input = latents_input.to(device)

                    in_noise = model.transformer(
                        hidden_states=latents_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds_ours_cp,
                        pooled_projections=pooled_prompt_embeds_ours_cp,
                        joint_attention_kwargs=model.joint_attention_kwargs,
                        return_dict=False,
                    )
                    in_noise = in_noise[0]

                if if_shuffle:
                    print('shuffle!')
                elif tranin_mode == 'real':
                    #last_latent_itr = start_latent
                    total_itr_loss = 0
                    for itr in range(num_real_itr):

                        with torch.enable_grad() if (if_train) else torch.no_grad():
                            #last_noise = in_noise
                            in_noise = in_noise.to(device4)
                            latents_input = latents_input.to(device4)
                            prompt_embeds_ours = prompt_embeds_ours.to(device4)
                            pooled_prompt_embeds_ours = pooled_prompt_embeds_ours.to(device4)
                            timestep = timestep.to(device4)
                            noise_pred,_ = s_net(
                                hidden_states=in_noise,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds_ours,
                                pooled_projections=pooled_prompt_embeds_ours,
                                en_hds_img=latents_input,
                                pool_hds_img=or_latent_db,
                                if_traning = if_train,
                                num_itr = 1
                            )

                            noise_pred_final = noise_pred
                            if model.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred_final = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            noise_pred_final = noise_pred_final.to(device)
                            latent_as_condition = itr_scheduler.step(-noise_pred_final, t, start_latent, return_dict=False,
                                                                     if_inv = True,if_last = itr==(num_real_itr-1))[0]

                            latents_input_itr = torch.cat([latent_as_condition] * 2) if model.do_classifier_free_guidance else latent_as_condition
                        if if_train:
                            #gt_latents_idx
                            if train_with_add and i < num_inference_steps*gt_latents_idx:

                                now_gt_noise = gt_latents[int(i*int(50/num_inference_steps))].to(noise_pred_final.device)
                                with torch.enable_grad() if if_train else torch.no_grad():
                                    kl_loss = F.kl_div((noise_pred_final).softmax(dim=-1).log(), (now_gt_noise).softmax(dim=-1),reduction = "batchmean")                                    
                                    #loss_qua = (loss_function_l1(noise_pred_final,now_gt_noise)+0.005*kl_loss)
                                    loss_qua = (loss_function_l2(noise_pred_final,now_gt_noise))
                                    loss_with_or = loss_function_l1(latent_as_condition.to(or_latent.device),or_latent)
                                    loss_with_last = loss_with_or if i == 0 else loss_function_l1(latent_as_condition.to(last_latent.device),last_latent)

                                    loss = loss_qua

                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                    total_itr_loss += loss.detach().item()
                                    torch.cuda.empty_cache()
                                    total_t_loss += total_itr_loss
                                    break
                            else:

                                prompt_embeds_ours_cp = prompt_embeds_ours.clone().to(device)
                                latents_input_itr = latents_input_itr.to(device)
                                pooled_prompt_embeds_ours_cp = pooled_prompt_embeds_ours.clone().to(device)
                                timestep = timestep.to(device)
                                with torch.no_grad():
                                    denoised_noise = model.transformer(
                                        hidden_states=latents_input_itr,
                                        timestep=timestep,
                                        encoder_hidden_states=prompt_embeds_ours_cp,
                                        pooled_projections=pooled_prompt_embeds_ours_cp,
                                        joint_attention_kwargs=model.joint_attention_kwargs,
                                        return_dict=False,
                                    )
                                    denoised_noise = denoised_noise[0]

                                with torch.enable_grad() if if_train else torch.no_grad():
                                    noise_pred = torch.nan_to_num(noise_pred.to(device4))
                                    denoised_noise = torch.nan_to_num(denoised_noise.to(device4))

                                    if itr_idx > 0:
                                        now_gt_noise = gt_latents[int(i*int(50/num_inference_steps))].to(noise_pred_final.device)
                                        loss_qua = (loss_function_l2(noise_pred_final,now_gt_noise)).to(device4)
                                        loss = 0.7*loss_function_l2(noise_pred,denoised_noise) + 0.3*loss_qua
                                    else:
                                        loss = loss_function_l2(noise_pred,denoised_noise)
                                    
                                    if not torch.isinf(loss):
                                        if if_train:
                                            if train_snet:
                                                optimizer.zero_grad()
                                                loss.backward()
                                                optimizer.step()
                                            else:
                                                loss.backward()

                                            if train_encoder:
                                                optimizer_emb.zero_grad()
                                                optimizer_emb.step()

                                            total_itr_loss += loss.detach().item()

                                            torch.cuda.empty_cache()
                                    else:
                                        print('inf!')
                                        minors += 1
                                        torch.cuda.empty_cache()

                            total_t_loss += total_itr_loss/(num_real_itr-minors)

                if not if_shuffle:
                    if model.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_final = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred_final = noise_pred_final.to(device)
                    start_latent = start_latent.to(device)
                    latent_as_condition = model.scheduler.step(-noise_pred_final, t, start_latent, return_dict=False,if_inv = True)[0]
                    latent_as_condition = latent_as_condition.to(device4)
                    if (inject_steps + inject_len)*num_inference_steps > i > inject_steps*num_inference_steps:
                        if i > 0:
                            latent_as_condition = 0.9*latent_as_condition + 0.1*last_latent.to(latent_as_condition.device)
                    last_latent = latent_as_condition
                    noised_latent_list.append(latent_as_condition.detach().cpu())

            if if_train:
                if tranin_mode == 'real':
                    total_epoch_loss += (total_t_loss/(num_inference_steps-num_rand_noise))
                    total_epoch_reg_loss += (total_t_reg_loss/(num_inference_steps-num_rand_noise))
                else:
                    total_epoch_loss += (total_t_loss/(num_inference_steps-num_rand_noise))
                #out_string = 'step '+str(epoch+1)+' avg loss in pass '+str(gap_output)+' steps is '+str(total_epoch_loss/gap_output)
                if np.isnan(total_epoch_loss):
                    print('total_epoch_loss: ',total_epoch_loss)
                    print('total_t_loss: ',total_t_loss)
                    print('(num_inference_steps-num_rand_noise): ',(num_inference_steps-num_rand_noise))
                    break
                out_string = 'loss of step ' + str(epoch+1) + ' is ' + str(total_epoch_loss)
                #+ ' and REG loss is ' + str(total_epoch_reg_loss)
                if tranin_mode == 'gen':
                    out_string = out_string + ', ' + type_str
                print(out_string)
                total_epoch_loss = 0
                total_epoch_reg_loss = 0
                with open(out_img_path+"save_log_"+ str(num_layers) +"_real_img.txt","a") as f:
                    f.write(out_string+'\n')
                if (epoch+1)%gap_output == 0 or epoch == 0:
                    num_inference_steps = 50
                    if (if_save and gap_output > 1) or (epoch+1)%10 == 0:
                        if train_snet and epoch > 0:
                            torch.save(s_net, model_path)
                            print('saved snet!')
                        model.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
                    with torch.no_grad():
                        real_img = real_img[0]
                        if tranin_mode == 'real':
                            real_img = out_img
                            prompt_c = prompt
                            #negative_prompt = ['']
                        #print('length of noised_latent_list:',len(noised_latent_list))
                        sd3_device = model.device
                        img = SD3(model,num_inference_steps=num_inference_steps,in_latents=latent_as_condition,if_train=if_train,real_img_idx = real_img_idx,
                            guidance_scale = guidance_scale,generator=generator,our_method=False,noised_latent_list = noised_latent_list,add_weight = add_weight,
                            prompt_embeds=prompt_embeds_ours_vis.to(sd3_device),pooled_prompt_embeds=pooled_prompt_embeds_ours_vis.to(sd3_device),out_data_path = out_data_path)
                        img = img[0]
                        img.save(out_img_path+'epoch'+str(epoch)+'.jpg')
                        real_img.save(out_img_path+'real_epoch_'+str(epoch)+'.jpg')
                if train_snet:
                    scheduler.step()
                if train_encoder:
                    scheduler_emb.step()
                progress_bar.update()
                break
            else:
                # not here!
                num_inference_steps = 50
                with torch.no_grad():
                    real_img = real_img[0]
                    if tranin_mode == 'real':
                        real_img = out_img
                        #prompt_c = prompt
                        prompt_c = ['']*batch_size
                    sd3_device = model.device
                    #print('latent_as_condition: ',latent_as_condition.shape)
                    img = SD3(model,num_inference_steps=num_inference_steps,in_latents=latent_as_condition,if_train=if_train,real_img_idx = real_img_idx,
                            guidance_scale = guidance_scale,generator=generator,our_method=False,noised_latent_list = noised_latent_list,add_weight = add_weight,
                            prompt_embeds=prompt_embeds_ours_vis.to(sd3_device),pooled_prompt_embeds=pooled_prompt_embeds_ours_vis.to(sd3_device),out_data_path = out_data_path)
                    img = img[0]

                    end_time = time.perf_counter()
                    print('DeepInv takes %ss' % ((end_time - start_time)))
                    #break

                    if if_tst and if_COCO:
                        img.save(out_tst_path + 'epoch'+str(epoch)+'.jpg')
                        real_img.save(out_tst_path + 'real_epoch'+str(epoch)+'.jpg')
                    elif if_tst and not if_COCO:
                        out_path_UE = './UltraEdit/inversed_results/'
                        img.save(out_path_UE + 'epoch'+str(epoch)+'.jpg')
                        real_img.save(out_path_UE + 'real_epoch'+str(epoch)+'.jpg')
                        torch.save(latent_as_condition.detach().cpu(),out_path_UE+str(epoch)+'.pt')
                    else:
                        img.save(out_tst_path + 'epoch'+str(epoch)+'.jpg')
                        real_img.save(out_tst_path + 'real_epoch'+str(epoch)+'.jpg')

            
    latents = latent_as_condition
    # Offload all models
    model.maybe_free_model_hooks()

    return latents,s_net
