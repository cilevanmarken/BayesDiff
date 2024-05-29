import argparse
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Embed images using CLIP model')
    parser.add_argument('--test_folder', type=str, required=True, help='Folder with test images to embed')
    parser.add_argument('--variant_folder', type=str, required=True, help='Folder with variant images to embed')
    parser.add_argument('--max_count', type=int, default=10000000, help='Maximum number of images to embed')
    # parser.add_argument('--image_number', type=int, default=1000000, help='Base number in the filenames')
    # parser.add_argument('--test_number', type=int, default=180001, help='Base number in the test filenames')
    return parser.parse_args()

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used is {device}')
    return device

def load_model_and_processor(device):
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("Loaded model and processor for vit-large-patch14")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_and_preprocess_image(image_path, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        return inputs
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def save_embeddings(embeddings, folder, model_name):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f'{model_name}_embedding.pt')
    torch.save(embeddings, file_path)
    print(f"Saved features to {file_path}")

def embed_images(image_folder, output_folder, model, processor, device, base_number, max_count, embed_size, model_name):
    # Check if embeddings already exist
    embeddings_file_path = os.path.join(output_folder, f'{model_name}_embedding.pt')
    if os.path.exists(embeddings_file_path):
        print(f"Embeddings already exist at {embeddings_file_path}. Skipping embedding process.")
        return

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # Sort filenames to process them in order
    n_images = min(len(image_files), max_count)

    if n_images == 0:
        print(f"No images found in {image_folder}.")
        return

    embeddings = torch.zeros((n_images, embed_size)).to(device)
    print(f"Embedding {n_images} images...")

    for i, filename in enumerate(tqdm(image_files[:n_images])):
        image_path = os.path.join(image_folder, filename)
        inputs = load_and_preprocess_image(image_path, processor, device)
        if inputs is None:
            continue

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        if 0 <= i < n_images:
            embeddings[i, :] = image_features

    print(f"Embedding shape: {embeddings.shape}")
    save_embeddings(embeddings, output_folder, model_name)

def main():
    args = parse_arguments()
    device = setup_device()
    model, processor = load_model_and_processor(device)
    embed_size = 768  # Dimension of embeddings
    model_name = 'vit-large-patch14'

    # Embedding test images
    test_image_folder = os.path.join(args.test_folder, 'images')
    test_output_folder = os.path.join(args.test_folder, 'embeddings')
    embed_images(test_image_folder, test_output_folder, model, processor, device, args.test_number, args.max_count, embed_size, model_name)

    # Embedding variant images
    variant_image_folder = os.path.join(args.variant_folder, 'images')
    variant_output_folder = os.path.join(args.variant_folder, 'embeddings')
    embed_images(variant_image_folder, variant_output_folder, model, processor, device, args.max_count, embed_size, model_name)

if __name__ == "__main__":
    main()
