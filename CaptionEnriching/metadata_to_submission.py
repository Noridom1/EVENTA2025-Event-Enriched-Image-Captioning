import json
import os
# Path
metadata_path = 'files/query_results_14B_115.json'
submission_path = '../results_a100/base_submission.csv'

output_dir = 'files/'
# output_dir = '../results/truncated_105_111'

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'submission.csv')

# Load truncated captions
with open(metadata_path, 'r', encoding='utf-8') as f:
    truncated_data = json.load(f)

# Create a map from query_id to truncated_caption
caption_map = {str(item['query_id']): item['generated_caption'] for item in truncated_data}

# Open input and output files
with open(submission_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    # Read and write header
    header = fin.readline()
    fout.write(header)

    # Process each line
    for line in fin:
        parts = line.rstrip('\n').split(',')
        query_id = parts[0]
        if query_id in caption_map:
            parts[-1] = f'\"{caption_map[query_id].replace('"', '').replace("\n", " ")}\"'  # Make sure to keep quotes
        fout.write(','.join(parts) + '\n')
