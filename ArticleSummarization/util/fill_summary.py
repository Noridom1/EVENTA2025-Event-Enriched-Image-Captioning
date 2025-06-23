import json

query_path = '../CaptionEnriching/files/query_new.json'
summary_path = 'files/to_summarize_giant.json'

query = json.load(open(query_path, 'r', encoding='utf-8')) 
summary = json.load(open(summary_path, 'r', encoding='utf-8')) 


cnt = 0
for q in query:
    if q['summary_article_content'] == "":
        q['summary_article_content'] = summary[q['article_id_1']]['summarized_content']
        cnt += 1

with open(query_path, 'w', encoding='utf-8') as f:
    json.dump(query, f, ensure_ascii=False, indent=2)

print(f"Filled {cnt} summary article content")