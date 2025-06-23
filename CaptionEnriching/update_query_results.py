import json

old_query_result_path = 'files/query_results_14B_115.json'
new_query_result_path = 'files/query_results_new.json'
retrieval_path = 'files/giant_retrieval_results.json'
database_path = '../data/database/database.json'
image_art_map_path = '../article_retrieval/img_art_map.json'

old_result = json.load(open(old_query_result_path, 'r', encoding='utf-8'))
retrieval_result = json.load(open(retrieval_path, 'r', encoding='utf-8'))
database = json.load(open(database_path, 'r', encoding='utf-8'))
image_art_map = json.load(open(image_art_map_path, 'r', encoding='utf-8'))

cnt = 0
for query in old_result:
    query_id = query['query_id']
    retrieval = retrieval_result[query_id + ".jpg"]
    image1 = retrieval['retrieved_image_ids'][0].replace('.jpg', '')
    query['image_raw_caption']  = ""
    if image1 != query['retrieved_image_id1']:
        new_article_id1 = image_art_map[image1]['article_id']
        # print(new_article_id1)
        query['article_id_1']  = new_article_id1
        query['generated_caption']  = ""
        query['retrieved_image_id1']  = image1
        query['article_content']  = database[new_article_id1]['content']
        query['url'] = database[new_article_id1]['url']
        query['title'] = database[new_article_id1]['title']
        query['summary_article_content'] = ""
        query['article_len'] = 0
        query['image_raw_caption']  = ""
        query['web_caption']  = ""
        cnt += 1

print('Total number of different R1:', cnt)

with open(new_query_result_path, 'w', encoding='utf-8') as f:
    json.dump(old_result, f, indent=2)