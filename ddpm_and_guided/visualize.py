import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms
import torchvision.utils as tvu
import os
import importlib
from runners.diffusion import Diffusion
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from ddimUQ_utils import parse_args_and_config, inverse_data_transform, dict2namespace
import yaml
import torchvision.transforms.functional as TF

to_pil = transforms.ToPILImage()


def get_dev_x_from_z(dev, exp, N, config, device):

    # get n samples from z distribution
    z_list = []
    for _ in range(N):
        z_list.append(
            exp + torch.rand_like(exp) * dev
        )
    
    # decode z into x
    Z = torch.stack(z_list,dim = 0)
    X = inverse_data_transform(config, Z.to(device))
    var_x = torch.var(X, dim = 0)
    exp_x = torch.mean(X, dim=0)
    dev_x = (var_x)**0.5

    return var_x, exp_x, dev_x


def visualize_uncertainty(exp_dir, id):

    # clear cache
    torch.cuda.empty_cache() 

    # search for a device
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 

    # load config
    with open("configs\imagenet128_guided.yml", "r") as f:
        conf = yaml.safe_load(f)
    config = dict2namespace(conf)

    #get z
    z_dev_list = []
    z_exp_list = []

    # load directories
    os.makedirs(f'{exp_dir}/x_dev',exist_ok=True)

    # make uncertanty map for image
    z_var_i = torch.load(f'{exp_dir}/z_var/{id}.pth')
    z_exp_i = torch.load(f'{exp_dir}/z_exp/{id}.pth')

    TF.to_grayscale(TF.to_pil_image(z_var_i*100)).save(f'{exp_dir}/z_var/AAAH{id}.jpg')

    z_dev_i = torch.clamp(z_var_i,min=0)**0.5

    TF.to_grayscale(TF.to_pil_image(z_dev_i*100)).save(f'{exp_dir}/z_var/AAH{id}.jpg')

    # z_dev_list.append(z_dev_i)
    # z_exp_list.append(z_exp_i)

    # N = 10
    # for index in range(1):
    #     z_dev = z_dev_list[index]
    #     z_exp = z_exp_list[index]
    #     _, _, dev_x = get_dev_x_from_z(z_dev, z_exp, N, config, device)
    #     # tvu.save_image(dev_x*255,f'{exp_dir}/x_dev/{id}.jpg')
    #     img = TF.to_grayscale(TF.to_pil_image(dev_x*10))
    #     img.save(f'{exp_dir}/x_dev/{id}.jpg')


def main():

    # parse args
    path = r"C:\Users\cilev\Documents\DL2\BayesDiff\ddpm_and_guided\exp\IMAGENET128\ddim_fixed_class51_train%200_step50_S10"
    visualize_uncertainty(path, "1000004_1.0_1.0")


if __name__ == "__main__":
    main()
