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

to_pil = transforms.ToPILImage()

def load_model_from_config(config, ckpt, args, verbose=False):
    print(f"Loading model from {ckpt}")
    pt_model = torch.load(ckpt, map_location="cpu")
    if "global_step" in pt_model:
        print(f"Global Step: {pt_model['global_step']}")

    model = instantiate_from_config(config, args)

    model.eval()
    
    return model


def instantiate_from_config(config, args):

    diffusion = Diffusion(args, config, rank=args.device)
    if diffusion.config.model.model_type == "guided_diffusion":
        print('yes')
        model = GuidedDiffusion_Model(
            image_size=diffusion.config.model.image_size,
            in_channels=diffusion.config.model.in_channels,
            model_channels=diffusion.config.model.model_channels,
            out_channels=diffusion.config.model.out_channels,
            num_res_blocks=diffusion.config.model.num_res_blocks,
            attention_resolutions=diffusion.config.model.attention_resolutions,
            dropout=diffusion.config.model.dropout,
            channel_mult=diffusion.config.model.channel_mult,
            conv_resample=diffusion.config.model.conv_resample,
            dims=diffusion.config.model.dims,
            num_classes=diffusion.config.model.num_classes,
            use_checkpoint=diffusion.config.model.use_checkpoint,
            use_fp16=diffusion.config.model.use_fp16,
            num_heads=diffusion.config.model.num_heads,
            num_head_channels=diffusion.config.model.num_head_channels,
            num_heads_upsample=diffusion.config.model.num_heads_upsample,
            use_scale_shift_norm=diffusion.config.model.use_scale_shift_norm,
            resblock_updown=diffusion.config.model.resblock_updown,
            use_new_attention_order=diffusion.config.model.use_new_attention_order,
        )

    return model

    




def get_dev_x_from_z(dev,exp,N, config, device):
     #get n samples from z distribution
    z_list = []
    for _ in range(N):
        z_list.append(
            exp + torch.rand_like(exp) * dev
        )
    
    #### decode z into x
    Z = torch.stack(z_list,dim = 0)
    X = inverse_data_transform(config, Z.to(device))
    var_x = torch.var(X,dim = 0)
    exp_x = torch.mean(X,dim=0)
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

    # model = load_model_from_config(config, r"C:\Users\cilev\Documents\DL2\external_utils\DDPM\128x128_diffusion.pt", args)#.to(device)

    #get z
    z_dev_list = []
    z_exp_list = []

    # load directories
    # exp_dir = r"C:\Users\cilev\Documents\DL2\BayesDiff\sd\ddim_exp\skipUQ\cfg3.0_A person in a hot-dog costume lying in a tub filled with mayonaisse_train1000_step50_S10"
    os.makedirs(f'{exp_dir}/x_dev',exist_ok=True)

    # make uncertanty map per image


    # id = 1000000
    z_var_i = torch.load(f'{exp_dir}/z_var/{id}.pth')
    z_exp_i = torch.load(f'{exp_dir}/z_exp/{id}.pth')
    z_dev_i = torch.clamp(z_var_i,min=0)**0.5
    z_dev_list.append(z_dev_i)
    z_exp_list.append(z_exp_i)


    N = 10
    for index in range(1):
        z_dev = z_dev_list[index]
        z_exp = z_exp_list[index]
        var_x, exp_x, dev_x = get_dev_x_from_z(z_dev,z_exp,N, config, device)
        # tvu.save_image(dev_x*100,f'{exp_dir}/x_dev/{id}.jpg' )

    tvu.save_image(dev_x*100,f'{exp_dir}/x_dev/{id}.jpg' )

    # return dev_x*100


def main():

    # parse args
    path = r"C:\Users\cilev\Documents\DL2\BayesDiff\ddpm_and_guided\exp\IMAGENET128\ddim_fixed_class51_train%200_step50_S10"
    visualize_uncertainty(path, "1000004_1.0_1.0")


if __name__ == "__main__":
    main()
