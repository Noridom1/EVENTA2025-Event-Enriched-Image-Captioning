import argparse
from TokenExtractor import TokenExtractor
from dataset import CustomDataset
import os
from Comparator import Comparator

def main():

    parser = argparse.ArgumentParser(description="Extract features and save to .pkl")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images")
    parser.add_argument('--save_path', type=str, required=True, help="Output .pkl file path")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for feature extraction")
    parser.add_argument('--start_batch', type=int, default=0, help="The index of the starting batch")


    args = parser.parse_args()

    # model_path = 'pre-trained/R101-Token.pth'
    # image_dir = '../data/test/pub_images'
    # image_path_1 = '../data/tmp/kiss_cup_query.jpg'
    # image_path_2 = '../data/tmp/kisscup.jpg'
    # image_path_3 = '../data/tmp/kiss_cup_noise.jpg'

    model_path = args.model_path
    image_dir = args.image_dir
    save_path = args.save_path
    batch_size = args.batch_size
    num_workers = args.num_workers

    model = TokenExtractor(model_path=model_path)
    print("[INFO] Successfully load the model")

    dataset = CustomDataset(image_dir=image_dir, transform=model.transform)
    print("[INFO] Successfully load the dataset")
    
    model.extract_dataset(dataset=dataset, batch_size=batch_size, num_workers=num_workers, save_path=save_path)

    # feature_vectors, all_paths = model.extract_dataset(dataset=dataset, batch_size=2)
    # print(feature_vectors.shape)


if __name__ == "__main__":
    main()