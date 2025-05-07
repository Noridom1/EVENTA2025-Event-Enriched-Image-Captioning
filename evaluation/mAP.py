import json

def compute_average_precision(gt_id, retrieved_list):
    """
    Compute AP for a single query.
    """
    ap = 0.0
    for i, retrieved_id in enumerate(retrieved_list):
        if retrieved_id == gt_id:
            ap = 1.0 / (i + 1)  # precision at the rank where the gt appears
            break
    return ap  # 0.0 if gt not found

def compute_map_from_json(json_path):
    """
    Compute mAP given a JSON file with retrieval results.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    ap_list = []
    for query_id, info in data.items():
        gt_id = info['gt']
        retrieved_list = info['retrieved']
        ap = compute_average_precision(gt_id, retrieved_list)
        ap_list.append(ap)

    map_score = sum(ap_list) / len(ap_list) if ap_list else 0.0
    return map_score

# Example usage
json_file_path = 'results/final_retrieval_train.json'
map_score = compute_map_from_json(json_file_path)
print(f"mAP: {map_score:.4f}")
