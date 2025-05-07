import json
import csv
from collections import defaultdict

def csv_to_json(csv_filename, save_json_path=None):
    json_data = {}

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.pop(next(iter(row)))  # Use the first column as key
            json_data[key.removesuffix('.jpg')] = row

    if save_json_path:
        with open(save_json_path, mode='w') as f:
            json.dump(json_data, f, indent=4)
    
    return json_data

def process_result_gt(result_json, gt_json, save_json_path=None):
    final_result = defaultdict(dict)

    for img_idx in gt_json:
        final_result[img_idx]['gt'] = gt_json[img_idx]['retrieved_image_id']
        final_result[img_idx]['retrieved'] = [result_json[img_idx][f'retrieved_image_id{i}'] for i in range(1, 11)]

    with open(save_json_path, 'w') as f:
        json.dump(final_result, f, indent=4)
    
    return final_result

    
# Paths
result_path = 'results/trainset_retrieval_results.csv'
gt_path = 'data/train/gt_train.csv'

result_json = csv_to_json(result_path)  
gt_json = csv_to_json(gt_path)


process_result_gt(result_json, gt_json, save_json_path='results/final_retrieval_train.json')


