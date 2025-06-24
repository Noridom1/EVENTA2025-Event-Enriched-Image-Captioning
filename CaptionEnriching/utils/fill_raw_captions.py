import json

query_results_path = 'files/metadata_final.json'
raw_captions_path = 'files/raw_caption_final.json'

query_results = json.load(open(query_results_path, 'r', encoding='utf-8'))
raw_captions = json.load(open(raw_captions_path, 'r', encoding='utf-8'))

for query in query_results:
    if query['image_raw_caption'] != "":
        continue
    image_id1 = query['retrieved_image_id1']
    raw_caption = raw_captions.get(image_id1, "")
    if not raw_caption:
        print(query['query_id'], image_id1)
    query['image_raw_caption'] = raw_caption

with open(query_results_path, 'w', encoding='utf-8') as f:
    json.dump(query_results, f, ensure_ascii=False, indent=2)