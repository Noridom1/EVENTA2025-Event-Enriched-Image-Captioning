import json

# === Load both files ===
summary1_path = 'files/to_summarize_giant_official.json'
summary2_path = 'files/to_summarize_14B.json'
output_path = 'files/summarized_articles.json'

with open(summary1_path, 'r', encoding='utf-8') as f1, open(summary2_path, 'r', encoding='utf-8') as f2:
    dict1 = json.load(f1)
    dict2 = json.load(f2)

# === Merge dictionaries ===
merged_dict = {}
all_keys = set(dict1.keys()).union(dict2.keys())

for key in all_keys:
    val1 = dict1.get(key, {})
    val2 = dict2.get(key, {})

    if val1:  # non-empty dict
        merged_dict[key] = val1
    elif val2:
        merged_dict[key] = val2
    else:
        merged_dict[key] = {}

# === Count missing keys (still empty dicts) ===
missing_count = sum(1 for v in merged_dict.values() if not v)

# === Output results ===
print(f"Number of missing keys (still empty dict): {missing_count}")

# Optionally, save the merged result
with open(output_path, 'w') as fout:
    json.dump(merged_dict, fout, indent=2)
