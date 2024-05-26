"""
Embed images using CLIP model vit-large-patch14. Makes an embedding of the images in a folder and saves them to a folder in the same directory as the input folder.
"""

import argparse
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Embed images using CLIP model')
    parser.add_argument('--variant_folder', type=str, required=True, help='Folder with images to embed')
    parser.add_argument('--max_count', type=int, default=10000000, help='Maximum number of images to embed')
    parser.add_argument('--image_number', type=int, default=1000000, help='Base number in the filenames')
    return parser.parse_args()

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used is {device}')
    return device

def load_model_and_processor(device):
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model = model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("Loaded model and processor for vit-large-patch14")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_and_preprocess_image(image_path, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        return inputs.to(device)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def main():
    args = parse_arguments()
    device = setup_device()
    img_folder = f'{args.variant_folder}/images'
    image_number = args.image_number
    embed_size = 768  # Dimension of embeddings
    model_name = 'vit-large-patch14'

    if not os.path.exists(img_folder):
        print(f"Image folder {img_folder} does not exist.")
        return

    model, processor = load_model_and_processor(device)
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_images = min(len(image_files), args.max_count)
    
    features = torch.zeros((n_images, embed_size)).to(device)

    print(f"Embedding {n_images} images...")
    for i, filename in enumerate(tqdm(image_files[:n_images])):
        image_path = os.path.join(img_folder, filename)
        inputs = load_and_preprocess_image(image_path, processor, device)
        if inputs is None:
            continue

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        try:
            index = int(filename.replace('img_', '').split('.')[0]) - image_number
            if 0 <= index < n_images:
                features[index, :] = image_features
            else:
                print(f"Skipping image with out-of-range index: {filename}")
        except ValueError as ve:
            print(f"Error processing filename {filename}: {ve}")

    print(f"Embedding shape: {features.shape}")
    out_folder = os.path.join(args.variant_folder, 'embedding')
    print(f'Out_fodler {out_folder}')
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, f'{model_name}_embedding.pt')
    torch.save(features, out_file)
    print(f"Saved features to {out_file}")

if __name__ == "__main__":
    main()
