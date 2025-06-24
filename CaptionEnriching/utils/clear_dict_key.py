import json

path = 'files/metadata_final_nogencap.json'
key = 'generated_caption'

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for query in data:
    query[key] = ''

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)