# BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference
## Reproducing and extending on the BayesDiff Framework

### *Ivo Brink, Cile van Marken, Sebastiaan Snel, Liang Telkamp, Jesse Wiers*

*May 2024*

This repository contains the codebase for reproducing and extending on the BayesDiff framework, as introduced in this [paper](https://arxiv.org/abs/2310.11142) by Kou, Gan, Wang, Li, and Deng (2024).


## Installation
The following code can be used to create an environment and clone this repo from github. 

```shell
conda create --name BayesDiff python==3.8
conda activate BayesDiff
conda install pip
git clone https://github.com/cilevanmarken/BayesDiff.git
cd BayesDiff
pip install -r requirements.txt
```

## Overview (unfinished)
BayesDiff
.
|-- readme.md   # Description of the repo with relevant getting started info (used dependencies/conda envs)
|               # and should contain all information necessary to reproduce the project
|
|-- blogpost.md # Main deliverable: a blogpost style report
|
|-- src         # contains the main project files
|   |-- lib     # feel free to use any folder structure, keep things organized
|   |-- data
|   |-- scripts


## Setup
In order for the code to work, some additional files are reuired.
1) Download [imagenet 128x128 ckpt of Guided Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt) to `/src/ddpm_and_guided/models`
2) Download [Imagenet](https://www.image-net.org/download.php) to `...`.
3) Download the [CelebA ckpt](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view) to `/src/ddpm_and_guided/models`
4) Download the [CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and only put the deepest img_align_celeba folder into `/src/ddpm_and_guided/datasets`.


## Usage
Here you'll find the specifications about the commands or job files needed to reproduce our research.

### High vs low unceartainty images
In line with the paper, we have generated 80 images for ImageNet and CelebA. These images are sorted by their variance, from high to low. The command for generating these sorted images is the following:

The generations can be found in the folder 'exp'.

### Visualizing the uncertainty maps
For every image that is generated, the code visualizes the uncertainty. These uncertainty maps can be found in the 'exp' folder. To reproduce the uncertainty maps from the blogpost, the following command can be executed:

### Hyperparameter search
We have performed a hyperparameter search over the sigma noise and prior precision of the LLLA layer. To reproduce these findings, the following command can be executed:

### Evaluation
For evaluation there are multiple python scripts to execute. For CelebA, we have created a test dataset of 22000 images, which we placed in /src/
```shell
cd /src/evaluation

PATH_FOLDER = PATH/TO/FOLDER-WITH-THE-IMAGES-AND-VARIANCES
python -u aggregation.py \
    --input_dir $PATH_FOLDER 

python -u embed.py --variant_folder $PATH_FOLDER \
    --image_number 1000000 \ 
    --test_folder $TEST_FOLDER \
    --test_number 180001

python -u aggregation.py \
    --input_dir $PATH_FOLDER 

python -u cmmd.py \
    --ref_embed $TEST_FOLDER \
    --eval_folder $PATH_FOLDER 

python -u fid.py \
    --path1 $TEST_FOLDER \
    --path2 $PATH_FOLDER  \
    --aggregation sum patch_max segmentation_mean random 
```

### Plotting the distribution of the uncertainty
The blogpost contains the distribution of the uncertainty values per image. These distributions can be recreated using the following command:

The plots will be saved in the folder ...

### Hessian Free Laplacian
The Hessian Free Laplacian is computed by altering the custom_model.py of the authors. The results can be reproduced by 





