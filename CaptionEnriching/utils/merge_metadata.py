import json

# List of input file paths
input_files = ['files/metadata_part_0.json', 'files/metadata_part_1.json', 'files/metadata_part_2.json', 'files/metadata_part_3.json']
output_file = 'files/metadata_giant.json'
merged_list = []

# Read and extend the merged_list from each file
for file in input_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_list.extend(data)
        else:
            print(f"Warning: {file} does not contain a list")

# Save the merged list to a new file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_list, f, ensure_ascii=False, indent=2)

print(f"Merge complete. Output saved to {output_file}")
