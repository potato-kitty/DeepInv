# DeepInv
This is official implementation of our paper ["DeepInv: A Novel Self-supervised Learning  Approach for Fast and Accurate  Diffusion Inversion"](https://arxiv.org/abs/2601.01487).

## Model link
Download [DeepInv.zip](https://huggingface.co/potatocatty/DeepInv). The model is too big for github thus we upload it on HuggingFace.

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

## Use Our Codes
1. Replace scheduling_flow_match_euler_discrete.py to .../site-packages/diffusers/schedulers/
2. Run "python3 main_tst_real.py"
3. We train our solver by chosen and pre-processed images from MS-COCO 2017 dataset, you could use your own dataset, or you could creat the same dataset as we did following the instruction from our previous project [EasyInv](https://github.com/potato-kitty/EasyInv).

## Citation
If you feel this project to be useful, please cite this paper and star it! The bibtex citation of our paper us as following.
```bibtex
@article{zhang2026deepinv,
  title={DeepInv: A Novel Self-supervised Learning Approach for Fast and Accurate Diffusion Inversion},
  author={Zhang, Ziyue and Lin, Luxi and Hu, Xiaolin and Chang, Chao and Wang, HuaiXi and Zhou, Yiyi and Ji, Rongrong},
  journal={arXiv preprint arXiv:2601.01487},
  year={2026}
}
```
