import time
import json
import logging
import argparse
import torch
import psutil
from torch.utils.data import DataLoader
from dataset import ArticleDataset
from summarizer import Summarizer
from tqdm import tqdm


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def log_resource_usage():
    # CPU memory
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    logging.debug(f"🧠 Memory Usage: {memory_mb:.2f} MB")

    # GPU (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        logging.debug(f"🟩 GPU Memory Allocated: {gpu_memory:.2f} MB")
        logging.debug(f"GPU Device: {torch.cuda.get_device_name()}")
    else:
        logging.debug("🟥 GPU not available")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize articles with a pre-trained model")
    parser.add_argument('--database', type=str, default='../data/database/database.json', help='Path to the database JSON')
    parser.add_argument('--to_summarize', type=str, default='files/to_summarize.json', help='Path to the to_summarize JSON')
    parser.add_argument('--log', type=str, default='logs/summarization.log', help='Path to log file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for summarization')
    parser.add_argument('--save_interval', type=int, default=10, help='Number of articles after which to save')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log)

    logging.info("🚀 Starting summarization job...")
    start_time = time.time()

    # Load dataset
    dataset = ArticleDataset(
        database_path=args.database,
        to_summarize_path=args.to_summarize,
        content_key="content",
        summarize_key="summarized_content"
    )

    if len(dataset) == 0:
        logging.info("✅ All articles are already summarized!")
        return

    logging.info(f"📄 {len(dataset)} articles to summarize.")

    # Load summarizer
    summarizer = Summarizer()
    summarizer.load_model()
    logging.info("🤖 Summarizer model loaded.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load existing summaries
    with open(args.to_summarize, 'r', encoding='utf-8') as f:
        to_summarize_dict = json.load(f)

    save_counter = 0
    
    for article in tqdm(dataloader, desc="Summarizing Articles"):
        article_id = article['id'][0]
        text = article['text'][0]

        log_resource_usage()

        try:
            summary = summarizer.summarize(text)
        except Exception as e:
            logging.warning(f"❌ Error summarizing article {article_id}: {e}")
            continue

        to_summarize_dict[article_id]["summarized_content"] = summary
        save_counter += 1

        logging.info(f"✅ Summarized {article_id}")

        time.sleep(0.1)

        # Periodic save
        if save_counter % args.save_interval == 0:
            with open(args.to_summarize, 'w', encoding='utf-8') as f:
                json.dump(to_summarize_dict, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Progress saved after {save_counter} articles.")

    # Final save
    with open(args.to_summarize, 'w', encoding='utf-8') as f:
        json.dump(to_summarize_dict, f, indent=2, ensure_ascii=False)
    logging.info("🎉 All summaries saved.")

    duration = time.time() - start_time
    logging.info(f"⏱️ Finished in {duration / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
