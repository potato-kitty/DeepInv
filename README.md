# DeepInv
This is official implementation of our paper "DeepInv: A Novel Self-supervised Learning  Approach for Fast and Accurate  Diffusion Inversion"

## Model link
https://huggingface.co/potatocatty/DeepInv. The model is too big for github thus we upload it on HuggingFace

## Simple use case
1. load model by "model = torch.load(model_path)"
2. predict inversion noise for one timestep
    "noise_pred,_ = model(
        hidden_states=DDIM_INV_noise,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        en_hds_img=latent_zt,
        pool_hds_img=latent_z0,
        if_traning = if_train,
        num_itr = 1
    )"
   Here, latent_z0 is the original input image edcoded by model's VAE, while latent_zt is the latent of current timestep.

## To be continue
Currently we only provide the parameters of SD3-inversion-solver, which we used in our paer for experiments. We will update full codes here soon.
