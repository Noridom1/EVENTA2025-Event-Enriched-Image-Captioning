import json
import json
import re


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def num_words(caption_text):
    return len(caption_text.split(' '))

def is_incompleted(caption_text):
    return (
        "<think>" in caption_text or
        num_words(caption_text) < 90 or
        # num_words(caption_text) > 300
        is_Chinese(caption_text)
    )

def is_Chinese(caption_text):
    return re.search(r'[\u4e00-\u9fff]', caption_text) is not None

def get_incompleted(results_path, thr=300):
    results = load_json(results_path)
    cnt = 0
    for query in results:
        caption = query.get("generated_caption", "")
        if is_Chinese(caption):
            cnt += 1
            print(query.get("query_id", "unknown"))

# results_path = 'files/query_results_14B_115.json'
# get_incompleted(results_path)
# caption_statistics(results_path, histogram=True)