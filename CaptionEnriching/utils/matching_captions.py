import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

# === Paths ===
metadata_path = 'files/metadata_final.json'
web_caption_pairs_path = '../DataProcessing/images_final/image_caption_pairs.json'
web_images_base_path = '../DataProcessing/images_final'
database_images_base_path = '../data/test/R1_final/'
output_path = 'files/metadata_final_captions.json'

# === Load JSONs ===
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print('len metadata:', len(metadata))
with open(web_caption_pairs_path, 'r', encoding='utf-8') as f:
    web_caption_pairs = json.load(f)

# === Load CLIP model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Count parameters
total_params = 0
print(f"{'Layer':70} {'Parameters'}")
print("="*90)
for name, param in model.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        total_params += num_params
        print(f"{name:70} {num_params}")
print("="*90)
print(f"{'Total Trainable Parameters':70} {total_params}")

# === Image preprocessing ===
def preprocess(image):
    return processor(images=image, return_tensors="pt")['pixel_values'][0]

# === Compute cosine similarity ===
def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1 / tensor1.norm()
    tensor2 = tensor2 / tensor2.norm()
    return torch.dot(tensor1, tensor2).item()

# === Main logic ===
num_missing_captions = 0
for query in tqdm(metadata):

    query_id = query['query_id']
    retrieve_image_id1 = query['retrieved_image_id1']

    if 'web_caption' in query and query['web_caption'] != "":
        continue

    # Load the reference image from database
    db_img_path = os.path.join(database_images_base_path, f"{retrieve_image_id1}.jpg")
    # db_img_path = os.path.join(database_images_base_path, f"{query_id}.jpg")


    if not os.path.exists(db_img_path):
        print(f"Missing DB image: {db_img_path}")
        continue
    db_img = Image.open(db_img_path).convert('RGB')
    db_img_tensor = preprocess(db_img).unsqueeze(0).to(device)

    # Get the corresponding web images and captions
    web_folder_path = os.path.join(web_images_base_path, query_id)
    if query_id not in web_caption_pairs:
        print(f"No captions for query {query_id}")
        continue

    web_images = web_caption_pairs[query_id]

    # Find best matching image from web using cosine similarity
    best_sim = -1
    best_caption = None

    for img_name, caption in web_images.items():
        web_img_path = os.path.join(web_folder_path, img_name)
        if not os.path.exists(web_img_path):
            print(f"Missing Web image: {web_img_path}")
            continue

        try:
            web_img = Image.open(web_img_path).convert('RGB')
            web_img_tensor = preprocess(web_img).unsqueeze(0).to(device)

            # Get embeddings
            with torch.no_grad():
                db_feat = model.get_image_features(db_img_tensor)
                web_feat = model.get_image_features(web_img_tensor)

            sim = cosine_similarity(db_feat[0], web_feat[0])
            if sim > best_sim:
                best_sim = sim
                best_caption = caption
        except Exception as e:
            print(f"Error processing {web_img_path}: {e}")
            continue

    # Save the best caption
    query['web_caption'] = best_caption if best_caption else ""
    if query['web_caption'] == "":
        num_missing_captions += 1

print("number of missing captions:", num_missing_captions)
# === Save updated metadata ===
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
