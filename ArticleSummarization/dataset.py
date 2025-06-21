import json
from torch.utils.data import Dataset

def num_words(text):
    return len(text.split(' '))

class ArticleDataset(Dataset):
    def __init__(self, database_path, to_summarize_path, content_key="content", summarize_key="summarized_content"):
        """
        :param database_path: Path to database.json
        :param to_summarize_path: Path to to_summarize.json (values contain summarize_key)
        :param content_key: Key for article body in database.json
        :param summarize_key: Key for storing the summary in to_summarize.json
        """
        with open(database_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)

        with open(to_summarize_path, 'r', encoding='utf-8') as f:
            self.to_summarize = json.load(f)

        self.article_ids = list(self.to_summarize.keys())

        self.articles = []
        summarized_count = 0
        unsummarized_count = 0

        for article_id in self.article_ids:
            article_data = self.database.get(article_id, {})
            summary_data = self.to_summarize.get(article_id, {})

            # Check if article exists and has the content key
            if content_key not in article_data:
                continue

            # Determine if the summary is empty/missing/null
            summary = summary_data.get(summarize_key, None)
            if summary in [None, "", "null"] or num_words(summary) < 100 or '<think>' in summary:
                self.articles.append({
                    "id": article_id,
                    "text": article_data[content_key]
                })
                unsummarized_count += 1
            else:
                summarized_count += 1

        print(f"✅ Already summarized: {summarized_count}")
        print(f"📝 To be summarized: {unsummarized_count}")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.articles[idx]
