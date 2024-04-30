import torch
from matplotlib import pyplot as plt
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torchvision import transforms
import torchvision.utils as tvu
import os

to_pil = transforms.ToPILImage()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def get_dev_x_from_z(dev,exp,N):
     #get n samples from z distribution
    z_list = []
    for _ in range(N):
        z_list.append(
            exp + torch.rand_like(exp) * dev
        )
    
    #### decode z into x
    Z = torch.stack(z_list,dim = 0)
    X = model.decode_first_stage(Z.to(device))
    var_x = torch.var(X,dim = 0)
    exp_x = torch.mean(X,dim=0)
    dev_x = (var_x)**0.5
    return dev_x


# search for a device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = OmegaConf.load(r"configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, r"C:\Users\cilev\Documents\DL2\external_utils\SD\v1-5-pruned-emaonly.ckpt").to(device)

#get z
z_dev_list = []
z_exp_list = []

# load directories
exp_dir = r"C:\Users\cilev\Documents\DL2\BayesDiff\sd\ddim_exp\skipUQ\cfg3.0_A person in a hot-dog costume lying in a tub filled with mayonaisse_train1000_step50_S10"
os.makedirs(f'{exp_dir}/x_dev',exist_ok=True)

# make uncertanty map per image


id = 1000000
z_var_i = torch.load(f'{exp_dir}/z_var/{id}.pth')
z_exp_i = torch.load(f'{exp_dir}/z_exp/{id}.pth')
z_dev_i = torch.clamp(z_var_i,min=0)**0.5
z_dev_list.append(z_dev_i)
z_exp_list.append(z_exp_i)






N = 10
for index in range(1):
    z_dev = z_dev_list[index]
    z_exp = z_exp_list[index]
    dev_x = get_dev_x_from_z(z_dev,z_exp,N)
    tvu.save_image(dev_x*100,f'{exp_dir}/x_dev/{id}.jpg' )