import argparse
import os
import json
import random
import shutil
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Constants
_SIGMA = 10
_SCALE = 1000

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--ref_embed', type=str, required=True, help='Embeddings of reference images')
    parser.add_argument('--eval_folder', type=str, required=True, help='Folder containing generated images and embeddings')
    parser.add_argument('--random', type=bool, default=True, help='Generate scores for random')
    parser.add_argument('--n_random', type=int, default=4400, help='How many images to use for random scores')
    return parser.parse_args()

def mmd(x, y):
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of embeddings."""
    x, y = x.to(device), y.to(device)
    assert x.shape == y.shape, 'Shapes of x and y are not the same'

    gamma = 1 / (2 * _SIGMA**2)
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))
    
    k_xx = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0))))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0))))
    
    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def get_n_samples(embedding, n, seed=42):
    """Get n random samples from embeddings."""
    random.seed(seed)
    idx = random.sample(range(embedding.shape[0]), n)
    return embedding[idx, :]

def load_and_evaluate(ref_embed_path, eval_folder_path, random=True, n_random=4400):
    """Load embeddings and evaluate MMD."""
    try:
        test_embeddings_path = os.path.join(ref_embed_path, 'embeddings', 'vit-large-patch14_embedding.pt')
        test_embeddings = torch.load(test_embeddings_path, map_location=device)
    except Exception as e:
        print(f"Error loading test embeddings: {e}")
        return

    try:
        gen_embeddings_path = os.path.join(eval_folder_path, 'embeddings', 'vit-large-patch14_embedding.pt')
        gen_embeddings = torch.load(gen_embeddings_path, map_location=device)
    except Exception as e:
        print(f"Error loading generated embeddings: {e}")
        return
    
    test_embeddings_sampled = get_n_samples(test_embeddings, gen_embeddings.shape[0])
    cmmd_normal = mmd(test_embeddings_sampled, gen_embeddings)
    # Get dir of eval_folder_path
    eval_folder_path = os.path.basename(eval_folder_path)

    print(f'CMMD distance for {eval_folder_path}: {cmmd_normal:.4f}')
    
    for agg in ['sum', 'patch_max', 'segmentation_mean']:
        cmmd_score = evaluate_aggregation(test_embeddings, gen_embeddings, eval_folder_path, agg)
        if cmmd_score is not None:
            print(f'CMMD distance for {eval_folder_path} {agg} excluded: {cmmd_score:.4f}')

    if random:
        random_scores = []
        for i in range(10):
            gen_embeddings_excluded = get_n_samples(gen_embeddings, n_random, seed=i)
            test_embeddings_excluded = get_n_samples(test_embeddings, n_random, seed=i)
            cmmd_score = mmd(test_embeddings_excluded, gen_embeddings_excluded)
            random_scores.append(cmmd_score)
            print(f'CMMD distance for {eval_folder_path} random {i}: {cmmd_score:.4f}')
        
        mean_random_scores = torch.tensor(random_scores).mean().item()
        std_random_scores = torch.tensor(random_scores).std().item()
        print(f'Mean of random scores: {mean_random_scores:.4f}')
        print(f'Std of random scores: {std_random_scores:.4f}')

def evaluate_aggregation(test_embeddings, gen_embeddings, eval_folder, aggregation):
    """Evaluate a specific aggregation method."""
    json_path = os.path.join(eval_folder, f'{aggregation}.json')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None
    
    # Sort data
    data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    image_list = [i for i, (k, v) in enumerate(data.items()) if v]
    agg_gen_embeddings = gen_embeddings[image_list]
    test_embeddings_sampled = get_n_samples(test_embeddings, agg_gen_embeddings.shape[0])
    
    return mmd(test_embeddings_sampled, agg_gen_embeddings)

def main():
    """Main function to run the evaluation."""
    args = parse_arguments()
    load_and_evaluate(args.ref_embed, args.eval_folder, random=args.random, n_random=args.n_random)

if __name__ == "__main__":
    main()
