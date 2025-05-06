import json
import re
from collections import defaultdict

images = defaultdict(list)
with open('database.json', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Remove control characters (ASCII < 32 except for newline and tab)
content = re.sub(r'(?<!\\)[\x00-\x1F]', ' ', content)
database = json.loads(content)
num_articles = len(database)
num_images = 0
i = 0
for article_id, article in database.items():
    image_ids = article['images']
    for image_id in image_ids:
        images[image_id].append(article_id)
        num_images += 1
    i += 1
    if i % 5000 == 0:
        print('Processed %d/%d (%.2f%% done)' % (i, num_articles, i*100.0/num_articles))


if num_images != len(images):
    print("????")
for image_id, articles in images.items():
    if len(articles) > 1:
        print(f"image_id: {image_id} has {len(articles)} articles: {articles}")

print("---DATABASE INFORMATION---")
print(f"[INFO] Number of articles: {num_articles}")
print(f"[INFO] Number of images: {num_images}")