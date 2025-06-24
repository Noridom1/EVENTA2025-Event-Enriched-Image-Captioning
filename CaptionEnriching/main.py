import time
import json
import logging
import argparse
import os
from tqdm import tqdm
from log import *
from caption_enricher import CaptionEnricher
from CaptionEnriching.utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Enrich image captions from query result")
    parser.add_argument('--model', type=str, default='Qwen3-14B', help='Model name')
    parser.add_argument('--json_file', type=str, default='files/query_results_1.json', help='Path to query JSON (read & write)')
    parser.add_argument('--log', type=str, default='logs/enrich_caption_1.log', help='Path to log file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for caption enriching')
    parser.add_argument('--save_interval', type=int, default=10, help='Number of queries after which to save progress')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log)

    logging.info("🚀 Starting Caption Enriching job...")
    start_time = time.time()

    # Load JSON as list of query dicts
    with open(args.json_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    logging.info(f"📄 Loaded {len(queries)} queries from {args.json_file}.")

    # Identify already-processed queries
    remaining_indices = [
        i for i, q in enumerate(queries) 
        if 'generated_caption' not in q or is_incompleted(q.get("generated_caption", ""))
    ]
    logging.info(f"📌 {len(remaining_indices)} queries to enrich (skipping {len(queries) - len(remaining_indices)}).")

    if not remaining_indices:
        logging.info("✅ All queries already enriched. Exiting.")
        return

    # Load model
    enricher = CaptionEnricher(model_name=args.model)
    enricher.load_model()
    logging.info("🤖 CaptionEnricher model loaded.")

    save_counter = 0
    for batch_idx in tqdm(range(0, len(remaining_indices), args.batch_size), desc="🔄 Enriching"):
        indices_batch = remaining_indices[batch_idx:batch_idx + args.batch_size]
        batch = [queries[i] for i in indices_batch]
        log_gpu_memory(f"Batch {batch_idx + 1}")

        try:
            # enriched_captions = enricher.enrich_caption_batch(batch)
            enriched_captions = enricher.enrich_caption_chunk_batch(batch)
        except Exception as e:
            logging.warning(f"❌ Error enriching batch {batch_idx + 1}: {e}")
            continue

        for idx, caption in zip(indices_batch, enriched_captions):
            queries[idx]['generated_caption'] = caption
            save_counter += 1
            logging.info(f"✅ Enriched caption for query ID: {queries[idx].get('query_id', '[unknown]')}")

        if save_counter % args.save_interval == 0:
            with open(args.json_file, 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Progress saved after {save_counter} enriched captions.")

        time.sleep(0.1)

    # Final save
    with open(args.json_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    logging.info("🎉 All enriched captions saved.")
    duration = time.time() - start_time
    logging.info(f"⏱️ Finished in {duration / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
