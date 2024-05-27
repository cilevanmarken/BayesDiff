import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import numpy as np
import os
import argparse
import json

# python aggregation_methods.py --input_dir /Users/liangtelkamp/Documents/master_ai/dl2/ivo
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
	device = torch.device('cpu')
	print('Using CPU')


def sum_score(var):
	return var.sum()


def patch_avg_max_score(var, patch_size=16):
	patches = var.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
	patches = patches.contiguous().view(-1, 16, 16)
	patch_avg = torch.mean(patches, dim=(1, 2))
	return patch_avg.max()


def get_foreground_masked_var(model, image, var):
	# Load a pre-trained model
	transform = T.Compose([
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	input_tensor = transform(image).unsqueeze(0)

	# Get the prediction mask
	with torch.no_grad():
		output = model(input_tensor)['out'][0]
	predicted_mask = output.argmax(0)

	# Convert to numpy and apply as mask
	mask = predicted_mask.byte().numpy()
	masked_var = np.where(mask, var, 0)
	return masked_var



def segmentation_sum_score(model, image, var):
	masked_var = get_foreground_masked_var(model, image, var)
	return masked_var.mean()



def get_all_info_dict(var_tensor_path, image_path, segmentation=False):
	info = {}
	for root, dirs, files in os.walk(var_tensor_path):
		for file in files:
			tensor = torch.load(os.path.join(var_tensor_path, file), map_location=torch.device('cpu'))
			tensor = tensor.mean(dim=0) # remove channels and average them

			# get name of image/variance
			name = file.replace(".pth", "").replace("var_", "")
			# ignore every character starting from first _
			name = name.split("_")[0]

			# save uncertainty (also image if segmentation is used)
			# save uncertainty (also image if segmentation is used)
			if not segmentation:
				info[name] = [tensor, None]
			else:
				exact_image_path = image_path + "/img_" + name + ".png"
				image = Image.open(exact_image_path)
				info[name] = [tensor, image]
	return info


def get_excluded_names(info, aggregation_method, segmentation=False):

	scores, name_score_dict = get_scores(info, aggregation_method, segmentation=segmentation)
	for name, score in name_score_dict.items():
		# True if score is below mean + std else False
		name_score_dict[name] = True if score < scores.mean() + scores.std() and score != 0 else False

	return name_score_dict


def get_scores(info, aggregation_method, segmentation=False):
	scores = []
	name_score_dict = {}

	if segmentation:
		model = deeplabv3_resnet50(pretrained=True)
		model.eval()

	for name, pair in info.items():
		var, image = pair

		if aggregation_method == "sum":
			score = sum_score(var)

		elif aggregation_method == "patch_max":
			#TODO: make patch_size different for smaller images?

			score = patch_avg_max_score(var)
		elif aggregation_method == "segmentation_sum":
			score = segmentation_sum_score(model, image, var)
		else:
			raise ValueError("Invalid aggregation method")

		scores.append(float(score))
		name_score_dict[name] = score
	return np.asarray(scores), name_score_dict


def main():
    parser = argparse.ArgumentParser(description='Process input directory.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing var_tensors and images folders')
    parser.add_argument('--out_dir', type=str, required=False, help='Input directory containing var_tensors and images folders')
    args = parser.parse_args()
	# If out_dir is not specified, save in the same directory
    if args.out_dir is None:
        args.out_dir = args.input_dir
	
    folder_name = os.path.basename(args.input_dir)

    print(f"Evaluation on {args.input_dir}")
    var_tensor_path = os.path.join(args.input_dir, "var_tensors")
    image_path = os.path.join(args.input_dir, "images")

    for aggregation_method in ["sum", "patch_max", "segmentation_sum"]:
        print(f"Aggregation_method: {aggregation_method}")
        segmentation = True if aggregation_method == "segmentation_sum" else False

        info = get_all_info_dict(var_tensor_path, image_path, segmentation=segmentation)

        name_score_dict = get_excluded_names(info, aggregation_method, segmentation=segmentation)

        # print(name_score_dict)
		# Save as json
        with open(f"{args.out_dir}/{aggregation_method}.json", "w") as f:
            json.dump(name_score_dict, f)

if __name__ == '__main__':
    main()