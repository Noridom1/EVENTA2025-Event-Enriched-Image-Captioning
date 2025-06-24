import json

# === Load the big JSON list ===
with open('files/metadata_final_nogencap.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Define custom sizes ===
sizes = [400, 400, 600, 600]
start_idx = 0

for i, size in enumerate(sizes):
    end_idx = start_idx + size
    chunk = data[start_idx:end_idx]

    # Save each chunk to a new file
    with open(f'files/metadata_new_temp_{i}.json', 'w', encoding='utf-8') as fout:
        json.dump(chunk, fout, indent=2)

    start_idx = end_idx

print("✅ Done. Split into [400, 400, 600, 600].")
