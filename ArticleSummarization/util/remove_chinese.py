import json

chinese_path = 'files/chinese_set.json'
to_summarize_path = 'files/to_summarize_dino_2.json'

def count_words(text):
    return len(text.split())

with open(chinese_path, 'r', encoding='utf-8') as f:
    chinese = json.load(f)
    
with open(to_summarize_path, 'r', encoding='utf-8') as f:
    to_summarize = json.load(f)

cnt = 0
for article_id in to_summarize:
    if count_words(to_summarize[article_id].get('summarized_content', "")) < 60:
        chinese.append(article_id)

for article_id in chinese:
    to_summarize[article_id] = {}
    cnt += 1

print("Number of resumarize:", cnt)

with open(to_summarize_path, 'w') as f:
    json.dump(to_summarize, f, indent=2)