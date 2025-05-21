import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(self.image_dir, image_name)
            for image_name in os.listdir(self.image_dir)
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path
    
    