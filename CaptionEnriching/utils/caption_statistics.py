import argparse
import matplotlib.pyplot as plt
from utils import *

def caption_statistics(metadata_path, histogram=False): 
    metadata = load_json(metadata_path)

    # Extract lengths of all generated captions
    caption_lengths = [num_words(result["generated_caption"]) for result in metadata if "generated_caption" in result]
    cnt = 0
    for result in metadata:
        if num_words(result['generated_caption']) < 50:
            print(result['query_id'])
        if is_Chinese(result['generated_caption']):
            print(result['query_id'])
            cnt += 1
        
    if not caption_lengths:
        print("No captions found.")
        return

    # Compute statistics
    mean_length = sum(caption_lengths) / len(caption_lengths)
    max_length = max(caption_lengths)
    min_length = min(caption_lengths)

    print(f"Caption Length Statistics:")
    print(f" - Mean: {mean_length:.2f}")
    print(f" - Max: {max_length}")
    print(f" - Min: {min_length}")
    print(f" - Total: {len(caption_lengths)}")
    print(f" - Chinese: {cnt}")

    # Plot histogram if requested
    if histogram:
        plt.hist(caption_lengths, bins=20, color='skyblue', edgecolor='black')
        plt.title("Histogram of Caption Lengths")
        plt.xlabel("Caption Length (words)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze caption length statistics from results JSON.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the results JSON file")
    parser.add_argument("--his", action="store_true", help="Show histogram of caption lengths")

    args = parser.parse_args()
    caption_statistics(args.metadata, args.his)

if __name__ == "__main__":
    main()
