import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from networks.RetrievalNet import Token
from PIL import Image
import time
import pickle
import os
from tqdm import tqdm

class TokenExtractor():
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.ms = [1, 2**(0.5), 2**(-0.5)]

    def _load_model(self, model_path):
        model = Token().to(self.device)
        state_dict = torch.load(model_path, map_location='cpu')
        load_result = model.load_state_dict(state_dict)
        # print(state_dict)
    
        # print("Missing keys:", load_result.missing_keys)
        # print("Unexpected keys:", load_result.unexpected_keys)
        return model
    
    @torch.no_grad()
    def extract_image(self, img_path):
        start = time.time()
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)
        feature =  self._extract_multi_scale(img)
        end = time.time()
        print(f'Extract the image in {end - start} seconds')
        return feature
    

    @torch.no_grad()
    def extract_dataset(self, dataset, batch_size=1, num_workers=0, save_progess=50, save_path=None):
    
        start = time.time()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        all_features = []
        all_paths = []
        cnt = 0

        for images, paths in tqdm(loader, desc='Extracting features', unit='batch'):

            images = images.to(self.device)
            vecs = self._extract_multi_scale(images)
            all_features.append(vecs.cpu().numpy())
            all_paths.extend(paths)
            cnt += 1

            # Save progress periodically
            if save_path is not None and cnt % save_progess == 0:
                feats = np.vstack(all_features)
                feature_dict = {
                    os.path.basename(path): feature for feature, path in zip(feats, all_paths)
                }
                self._save_features_to_file(feature_dict, save_path)
                print(f"[INFO] Saved after {cnt} batches")

        all_features = np.vstack(all_features)

        # Final save
        if save_path:
            feature_dict = {
                os.path.basename(path): feature for feature, path in zip(all_features, all_paths)
            }
            self._save_features_to_file(feature_dict, save_path)

        print(f'[INFO] Finished extracting features in {time.time() - start:.2f} seconds.')
        return all_features, all_paths



    def _extract_multi_scale(self, imgs):
        # imgs: (B, C, H, W)
        vecs = []
        for s in self.ms:
            size = [int(imgs.shape[2] * s), int(imgs.shape[3] * s)]
            resized_img = torch.nn.functional.interpolate(imgs, size=size, mode="bilinear", align_corners=False)
            feat_vec = self.model.forward_test(resized_img) # (1, D)
            feat_vec = torch.nn.functional.normalize(feat_vec, p=2, dim=1)
            vecs.append(feat_vec)
        
        vecs = torch.stack(vecs).mean(0) # average bool
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        
        return vecs
    
    def _save_features_to_file(self, feature_dict, filepath):

        with open(filepath, "wb") as f:
            pickle.dump(feature_dict, f)

        print(f"Saved {len(feature_dict)} features to {filepath}")

        
    
