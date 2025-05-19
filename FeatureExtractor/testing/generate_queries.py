
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset
from PIL import Image
import os
import argparse
import json
import random

transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.Resize((
        random.randint(600, 1200),  # height
        random.randint(600, 1200)   # width
    ))(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_test_set(image_dir, num_image, transform, save_dir):
    all_images = os.listdir(image_dir)
    selected_images = random_images(all_images, num_image)
    print(f"[INFO] Selected {num_image} images")

    images = load_images(image_dir, selected_images)
    print(f"[INFO] Loaded {num_image} images")

    transformed_images = apply_transform(images, transform)
    print(f"[INFO] Applied transformation")

    ground_truth = generate_gt(selected_images)
    print(f"[INFO] Generated ground_truth")

    save_images_path = os.path.join(save_dir, 'images')

    save_images(transformed_images, save_images_path)
    save_ground_truth(ground_truth, save_dir)
    print(f"[INFO] Saved small test set successfully")


def load_images(image_dir, selected_images):
    return [Image.open(os.path.join(image_dir, selected_image)) for selected_image in selected_images]

def random_images(all_images, num_images):
    if num_images > len(all_images):
        raise ValueError("num_images cannot be greater than the number of available images.")
    selected_images = random.sample(all_images, num_images)
    return selected_images

def apply_transform(images, transform):

    transformed_images = [transform(selected_image) for selected_image in images]
    return transformed_images

def generate_gt(selected_images):
    gt = {
        f"{index:04d}.jpg" : image_name for index, image_name in enumerate(selected_images)
    }
    return gt

def denormalize(tensor, mean, std):
    """
    Undo normalization for a tensor image.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def save_images(transformed_images, save_images_path):
    os.makedirs(save_images_path, exist_ok=True)
    to_pil = transforms.ToPILImage()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for idx, img in enumerate(transformed_images):
        filename = f"{idx:04d}.jpg"
        save_path = os.path.join(save_images_path, filename)

        # Denormalize, clamp to [0,1], and convert to PIL
        img = denormalize(img, mean, std).clamp(0, 1)
        pil_img = to_pil(img)
        pil_img.save(save_path, format="JPEG")


def save_ground_truth(ground_truth, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'gt.json'), 'w') as f:
        json.dump(ground_truth, f, indent=4)
    

def main():
    parser = argparse.ArgumentParser(description="Create a small test image set")
    
    parser.add_argument('--num_image', type=int, default=100,
                        help='Number of images to select randomly')

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to the directory containing all images')

    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to the directory where the test set will be saved')
    
    args = parser.parse_args()

    num_image = args.num_image
    image_dir = args.image_dir
    save_dir = args.save_dir

    generate_test_set(image_dir, num_image, transform=transform, save_dir=save_dir)

if __name__ == "__main__":
    main()
    
