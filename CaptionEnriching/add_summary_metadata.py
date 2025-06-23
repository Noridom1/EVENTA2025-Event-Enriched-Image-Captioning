import json
import statistics
import matplotlib.pyplot as plt

query_results_path = 'files/query_results_new.json'
summary_path = '../ArticleSummarization/files/to_summarize_14B.json'
save_path = 'files/query_new.json'
show_histogram = True  # Set to False if you don't want to show the histogram

# Load query results and summaries
with open(query_results_path, 'r', encoding='utf-8') as f:
    query_results = json.load(f)

with open(summary_path, 'r', encoding='utf-8') as f:
    summaries = json.load(f)

# Add summary and collect word lengths
summary_lengths = []
cnt_missing = 0

for result in query_results:
    article_id_1 = result['article_id_1']
    summary = summaries.get(article_id_1, "")
    if summary:
        summary = summary.get('summarized_content', "")
    result['summary_article_content'] = summary
    if summary == "":
        cnt_missing += 1
        continue
    word_count = len(summary.split())
    summary_lengths.append(word_count)

# Save the updated results
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(query_results, f, indent=2)

# Statistics
mean_len = statistics.mean(summary_lengths)
min_len = min(summary_lengths)
max_len = max(summary_lengths)

print(f"Summary Word Count Statistics:")
print(f"  Mean: {mean_len:.2f} words")
print(f"  Min : {min_len} words")
print(f"  Max : {max_len} words")
print(f"  Missing : {cnt_missing} articles")


# Histogram (if enabled)
if show_histogram:
    plt.figure(figsize=(10, 5))
    plt.hist(summary_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Summary Word Counts')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
