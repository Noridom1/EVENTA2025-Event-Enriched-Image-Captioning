import json

file_path = 'files/query_newraw.json'
new_raw_path = 'files/gemma_long.json'


with open(file_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

with open(new_raw_path, 'r', encoding='utf-8') as f:
    new_raw = json.load(f)

for query in result:
    retrieved_image_id1 = query.get('retrieved_image_id1', None)

    if not retrieved_image_id1:
        print('retrieved_image_id1 does not exist')
        break
    
    new_raw_cap = new_raw.get(retrieved_image_id1, None)
    if not new_raw_cap:
        print('new raw caption does not exist')
        break
    
    query['image_raw_caption'] = new_raw_cap
    

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
