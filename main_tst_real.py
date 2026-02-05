# %%
from diffusers import StableDiffusion3Pipeline
import torch
import utils
from model import *
from accelerate import init_empty_weights,load_checkpoint_and_dispatch,load_checkpoint_in_model,dispatch_model

#g_cpu = torch.Generator().manual_seed(1234)
g_cpu = None
NUM_DIFFUSION_STEPS = 50

def run_and_display(model,prompts,if_save = True, generator=None,num_round = 1,img_name=None,output_folder=None,if_train = False,order = 0,batch_size = 8,
                    epochs = 5000,condition = False, new_model = False, train_with_add = False,num_inference_steps = 50,itr_idx = 0,fr_round = False,num_layers = 5,
                    if_shuffle = True,device2 = None, tranin_mode = 'real',gap_output = 20,use_new_model = False):
    _ = utils.text2image_ldm_stable(model, prompt=prompts, num_inference_steps=num_inference_steps,if_shuffle=if_shuffle, device2= device2,tranin_mode = tranin_mode,num_layers = num_layers,
                                         generator=generator,num_round=num_round,if_train=if_train,order=order,batch_size=batch_size,train_with_add = train_with_add,use_new_model = use_new_model,
                                         epochs=epochs,if_save=if_save,gap_output = gap_output,condition = condition,new_model = new_model,itr_idx = itr_idx,fr_round = fr_round)

    return 0


max_memory={2: "24GiB", 3: "24GiB"}
ldm_stable = StableDiffusion3Pipeline.from_pretrained("./pretrained_models/sdv3-main",device_map = 'balanced',max_memory = max_memory,torch_dtype=torch.bfloat16)

batch_size = 4
prompts = ['']*batch_size

num_layers = 5
lr = 1e-5
use_new_model = True


for idx in range(5):
    itr_idx = idx
    if itr_idx == 0:
        train_with_add = False
    else:
        train_with_add = True
    fr_round = True

    out_path = './DeepInv/final_version/itr_' + str(itr_idx) + '/'
    for repeat in range(2):
        num_inference_steps_list = [1,5,10,25,50]
        if itr_idx == 4:
            if repeat == 0:
                num_inference_steps_list = [50]
                full_train = False
                use_new_model = True
            else:
                num_inference_steps_list = [50]
                full_train = True

        for inf_num_idx,num_inference_steps in enumerate(num_inference_steps_list):
            if num_inference_steps < 50 and fr_round:
                out_img_path_next = out_path + str(num_layers) + '_layers_with_' + str(num_inference_steps_list[inf_num_idx+1]) + '_steps/'
                if os.path.exists(out_img_path_next):
                    continue
            if num_inference_steps < 10:
                train_epoachs = 300
                gap_output = 50
            elif num_inference_steps < 25:
                train_epoachs = 250
                gap_output = 20
            elif num_inference_steps < 50:
                train_epoachs = 200
                gap_output = 10
            else:
                train_epoachs = 100
                gap_output = 5


            _ = utils.text2image_ldm_stable(ldm_stable,prompts,generator=g_cpu,if_train=True,if_save=True,epochs=train_epoachs,if_shuffle=False,gap_output = gap_output,itr_idx = itr_idx,num_layers = num_layers,full_train = full_train,batch_size = batch_size,
                                device2=torch.device('cuda:1'),tranin_mode = 'real',condition = False, new_model = True,train_with_add=train_with_add,num_inference_steps = num_inference_steps,fr_round = fr_round,use_new_model = use_new_model,lr = lr)

        if not train_with_add:
            use_new_model = False
            break
        else:
            fr_round = False

    
    num_inference_steps = 50
    batch_size = 1
    prompts = ['']*batch_size
    _ = utils.text2image_ldm_stable(ldm_stable,prompts,generator=g_cpu,if_train=False,if_save=False,epochs=300,if_shuffle=False,gap_output = 1,num_layers = num_layers,num_inference_steps = num_inference_steps,batch_size = batch_size,
                        device2=torch.device('cuda:1'),tranin_mode = 'real',condition = False, new_model = True,itr_idx=itr_idx)




