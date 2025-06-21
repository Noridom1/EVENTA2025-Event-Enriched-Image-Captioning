import csv
import json

def generate_to_summarize_from_train(csv_path):
    article_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            article_id = row['retrieved_article_id'].strip()
            if article_id:
                article_ids.add(article_id)
    return article_ids


def generate_to_summarize_from_retrieval(csv_path):
    article_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ids = [row[f'article_id_{i}'].strip() for i in range(1, 2)]
            for article_id in ids:
                if article_id:
                    article_ids.add(article_id)
    return article_ids


# Paths
gt_train_path = '../data/train/gt_train.csv'
retrieval_path = 'files/submission_dino.csv'
output_path = 'files/to_summarize.json'

# Generate ID sets
# article_ids_train = generate_to_summarize_from_train(gt_train_path)
# print("train:", len(article_ids_train))
article_ids_retrieval = generate_to_summarize_from_retrieval(retrieval_path)
print("retrieval:", len(article_ids_retrieval))

# Merge sets
all_article_ids = article_ids_retrieval
# all_article_ids = article_ids_train.union(article_ids_retrieval)

# Build output dict
to_summarize_dict = {article_id: {} for article_id in all_article_ids}

# Save to JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(to_summarize_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(all_article_ids)} article IDs to be summarized into '{output_path}'")
