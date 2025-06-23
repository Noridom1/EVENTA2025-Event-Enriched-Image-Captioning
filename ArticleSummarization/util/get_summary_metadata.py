import json

metadata_path = '../CaptionEnriching/files/query_results_new.json'
to_summarize_giant_path = 'files/to_summarize_giant_check.json'

# Load the metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    query_results = json.load(f)

# Filter for empty 'summary_article_content'
filtered_dict = {
    item["article_id_1"]: {}
    for item in query_results
    if item.get("summary_article_content", "") == ""
}

# Save to new JSON file
with open(to_summarize_giant_path, "w", encoding="utf-8") as f:
    json.dump(filtered_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(filtered_dict)} entries with empty summary.")
