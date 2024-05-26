import argparse
import os
import json
import shutil
import logging
from pytorch_fid.fid_score import calculate_fid_given_paths

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate FID between two sets of images')
    parser.add_argument('--path1', type=str, required=True, help='Path to the first set of images') # Path should go to folder where images are
    parser.add_argument('--path2', type=str, required=True, help='Path to the folder where the folder of images is') # Path should go to folder where the folder of images is
    parser.add_argument('--aggregation', type=str, default='all', required=True, help='Aggregation method')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_paths(path1, path2):
    if not os.path.exists(path1):
        logging.error(f"Path {path1} does not exist.")
        return False
    if not os.path.exists(path2):
        logging.error(f"Path {path2} does not exist.")
        return False
    return True

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {json_path}: {e}")
        return None

def create_temp_folder(base_path):
    temp_folder = os.path.join(base_path, 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    return temp_folder

def copy_images(image_list, src_folder, dst_folder):
    for image in image_list:
        src_path = os.path.join(src_folder, f'img_{image}.png')
        dst_path = os.path.join(dst_folder, f'img_{image}.png')
        try:
            shutil.copy(src_path, dst_path)
        except Exception as e:
            logging.error(f"Error copying image {src_path} to {dst_path}: {e}")

def calculate_fid_for_aggregation(path1, path2, aggregation, batch_size, device):
    json_path = os.path.join(path2, f'{aggregation}.json')
    sum_excluded = load_json(json_path)
    if sum_excluded is None:
        return None

    image_list = [k for k, v in sum_excluded.items() if v]
    print(f"Length of list for {aggregation} is {len(image_list)}")
    temp_folder = create_temp_folder(path2)

    try:
        copy_images(image_list, os.path.join(path2, 'images'), temp_folder)
        fid_value = calculate_fid_given_paths([path1, temp_folder], batch_size, device, 2048)
    finally:
        shutil.rmtree(temp_folder)
    
    return fid_value

def main():
    setup_logging()
    args = parse_arguments()

    path1 = os.path.join(args.path1, 'images')
    path2 = os.path.join(args.path2, 'images')
    if not validate_paths(path1, path2):
        return

    batch_size = args.batch_size
    device = args.device
    aggregation = args.aggregation

    if aggregation == "all":
        fid_value = calculate_fid_given_paths([path1, path2], batch_size, device, 2048)
    else:
        fid_value = calculate_fid_for_aggregation(path1, args.path2, aggregation, batch_size, device)

    if fid_value is not None:
        print(f"FID: {fid_value}")

if __name__ == "__main__":
    main()
