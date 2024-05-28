import argparse
import os
import json
import random
import shutil
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# Constants
_SIGMA = 10
_SCALE = 1000

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--ref_embed', type=str, required=True, help='Embeddings of reference images')
    parser.add_argument('--eval_folder', type=str, required=True, help='Folder containing generated images and embeddings')
    return parser.parse_args()

# Memory-efficient MMD implementation in PyTorch
def mmd(x, y):
    x, y = x.to(device), y.to(device)
    assert x.shape == y.shape, 'Shapes of x and y are not the same'

    gamma = 1 / (2 * _SIGMA**2)
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))
    
    k_xx = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0))))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))))
    
    return _SCALE * (k_xx + k_yy - 2 * k_xy)

# Get n random samples from embeddings
def get_n_samples(embedding, n, seed=42):
    random.seed(seed)
    idx = random.sample(range(embedding.shape[0]), n)
    return embedding[idx, :]

# Load embeddings and evaluate MMD
def load_and_evaluate(ref_embed_path, eval_folder_path):
    test_embeddings = torch.load(ref_embed_path, map_location=device)
    gen_embeddings_path = os.path.join(eval_folder_path, 'embedding', 'vit-large-patch14_embedding.pt')
    gen_embeddings = torch.load(gen_embeddings_path, map_location=device)
    
    test_embeddings_normal = get_n_samples(test_embeddings, gen_embeddings.shape[0])
    cmmd_normal = mmd(test_embeddings_normal, gen_embeddings)
    print(f'CMMD distance for {eval_folder_path}: {cmmd_normal:.4f}')
    
    for agg in ['sum', 'patch_max', 'segmentation']:
        cmmd_score = evaluate_aggregation(test_embeddings, gen_embeddings, eval_folder_path, agg)
        print(f'CMMD distance for {eval_folder_path} {agg} excluded: {cmmd_score:.4f}')

    # Randomly select 4400 gen embeddings and evaluate MMD
    random_scores = []
    for i in range(10):
        gen_embeddings_excluded = get_n_samples(gen_embeddings, 4400, seed=i)
        test_embeddings_excluded = get_n_samples(test_embeddings, 4400, seed=i)
        cmmd_score = mmd(test_embeddings_excluded, gen_embeddings_excluded)
        random_scores.append(cmmd_score)
        print(f'CMMD distance for {eval_folder_path} random {i}: {cmmd_score:.4f}')
    # Print mean and std of random scores
    print(f'Mean of random scores: {torch.tensor(random_scores).mean():.4f}')
    print(f'Std of random scores: {torch.tensor(random_scores).std():.4f}')

# Evaluate aggregation method
def evaluate_aggregation(test_embeddings, gen_embeddings, eval_folder, aggregation):
    json_path = os.path.join(eval_folder, f'{aggregation}.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_list = [int(k) - 1000000 for k, v in data.items() if v]
    agg_gen_embeddings = gen_embeddings[image_list]
    test_embeddings_excluded = get_n_samples(test_embeddings, agg_gen_embeddings.shape[0])
    
    return mmd(test_embeddings_excluded, agg_gen_embeddings)

# Main
def main():
    args = parse_arguments()
    load_and_evaluate(args.ref_embed, args.eval_folder)

if __name__ == "__main__":
    main()
