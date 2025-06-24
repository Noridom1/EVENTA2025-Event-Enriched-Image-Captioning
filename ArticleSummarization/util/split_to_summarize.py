import json

# === Load original JSON ===
to_summarize_path = 'files/to_summarize_magic.json'


with open(to_summarize_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Split into 4 parts ===
items = list(data.items())
chunk_size = len(items) // 4 + (len(items) % 4 > 0)  # ensures all items are included

for i in range(4):
    chunk = dict(items[i * chunk_size:(i + 1) * chunk_size])
    with open(f'files/to_summarize_magic_{i}.json', 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)
