import argparse
from TokenExtractor import TokenExtractor
from dataset import CustomDataset
import os
from Comparator import Comparator

def main():

    model_path = 'pre-trained/R101-Token.pth'
    image_dir = '../data/test/pub_images'
    image_path_1 = '../data/tmp/kiss_cup_query.jpg'
    image_path_2 = '../data/tmp/kisscup.jpg'
    image_path_3 = '../data/tmp/kiss_cup_noise.jpg'


    model = TokenExtractor(model_path=model_path)
    cmp = Comparator()
    print("[INFO] Successfully load the model")

    dataset = CustomDataset(image_dir=image_dir, transform=model.transform)

    
    print("[INFO] Successfully load the dataset")
    feat_vec1 = model.extract_image(image_path_1)
    feat_vec2 = model.extract_image(image_path_2)
    feat_vec3 = model.extract_image(image_path_3)

    print(f"Cosine similarity query and real: {cmp.cosine_similarity(feat_vec1, feat_vec2)}")
    print(f"Cosine similarity query and noise: {cmp.cosine_similarity(feat_vec1, feat_vec3)}")

    # feature_vectors, all_paths = model.extract_dataset(dataset=dataset, batch_size=2)
    # print(feature_vectors.shape)


if __name__ == "__main__":
    main()