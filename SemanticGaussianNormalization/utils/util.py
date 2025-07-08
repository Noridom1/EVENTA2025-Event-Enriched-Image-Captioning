from huggingface_hub import InferenceClient
import re
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
from typing import List, Union
import os
import zipfile
import shutil
import spacy
from tqdm import tqdm
import random
from collections import Counter
import math
from collections import Counter
import json
import streamlit as st
from PIL import Image
import os
import re
from glob import glob

import csv
import json

def create_retrieval_csv(
    retrieval_json_path,
    database_json_path,
    output_csv_path="/retrieval_submission.csv",
    top_k_articles=10
):
    """
    Create a CSV file with retrieval results formatted as:
    query_id, article_id_1, ..., article_id_10, generated_caption

    Args:
        retrieval_json_path (str): Path to JSON retrieval results (output).
        database_json_path (str): Path to JSON mapping of articles to images.
        output_csv_path (str): Path to output CSV file.
        top_k_articles (int): Number of unique top articles to include.
    """

    lorem_text = "against the event was between yellow its moment in a sense of the image depicts a moment of a where with a who from stark within serves as a emphasizing visual depicts surrounding not world they red two for the photograph captures the image captures a to the at the powerful presence while the which vibrant scene team and the likely their face white potential showcasing significant nature of this image captures impact are it black on the his highlights the her be 's as into emphasizes the or about colors by the suggests victory atmosphere highlighting the is expression significance in this photograph during blue an in the reminder of that sense he backdrop background image with the"

    lorem_text = '"' + lorem_text + '"'

    # Load retrieval results
    with open(retrieval_json_path, "r") as f:
        output = json.load(f)

    # Load article mapping
    with open(database_json_path, "r") as f:
        article_data = json.load(f)

    # Build reverse lookup: image_id.jpg -> article_id
    image_to_article = {}
    for article_id, info in article_data.items():
        for img_id in info["images"]:
            image_to_article[img_id + ".jpg"] = article_id

    # Prepare CSV rows
    csv_rows = []

    for query_key, retrieval in output.items():
        retrieved_images = retrieval["retrieved_image_ids"]

        # Collect unique articles in retrieval order
        seen_articles = set()
        top_articles = []
        for img in retrieved_images:
            # img might be e.g., "a23834558aad36e9.jpg"
            article_id = image_to_article.get(img)
            if article_id and article_id not in seen_articles:
                top_articles.append(article_id)
                seen_articles.add(article_id)
            if len(top_articles) == top_k_articles:
                break

        # Pad if needed
        while len(top_articles) < top_k_articles:
            top_articles.append("NA")

        # Build row
        query_id = query_key.replace(".jpg", "")
        row = [query_id] + top_articles + [f"{lorem_text}"]
        csv_rows.append(row)

    # Write CSV
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = (
            ["query_id"]
            + [f"article_id_{i}" for i in range(1, top_k_articles + 1)]
            + ["generated_caption"]
        )
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"CSV saved to {output_csv_path}")

def merge_retrieve_parts(folder_path, output_path="patch_rr_retrieve_top100.json"):
    # Find all JSON part files in the folder
    part_files = sorted(glob(os.path.join(folder_path, "patch_rr_retrieve_top100_*.json")))

    merged_dict = {}
    for part_file in part_files:
        with open(part_file, 'r', encoding='utf-8') as f:
            part_data = json.load(f)
            merged_dict.update(part_data)
        print(f"[🔗 Merged] {os.path.basename(part_file)} with {len(part_data)} items")

    # Save the merged dictionary
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_dict, f, indent=2, ensure_ascii=False)

    print(f"[✅ Done] Merged {len(part_files)} files → {output_path} ({len(merged_dict)} total queries)")


def split_retrieve_file(retrieve_path, split_unit=300, save_folder="retrieve_part"):
    # Load full retrieval dictionary
    with open(retrieve_path, 'r', encoding='utf-8') as f:
        retrieve_dict = json.load(f)

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    all_keys = list(retrieve_dict.keys())
    total = len(all_keys)

    # Split into chunks of split_unit size
    for idx in range(0, total, split_unit):
        chunk_keys = all_keys[idx:idx + split_unit]
        split_dict = {k: retrieve_dict[k] for k in chunk_keys}
        out_path = os.path.join(save_folder, f"retrieve_part_1_{(idx // split_unit) + 1}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2, ensure_ascii=False)
        print(f"[✅ Saved] Part {(idx // split_unit) + 1}: {len(chunk_keys)} queries → {out_path}")


def getR100Json(retrieve_path, save_path="R100_list.json"):
    data = read_json(retrieve_path)
    r100_set = set()

    for query_id, result in data.items():
        top100_ids = result.get("retrieved_image_ids", [])[:100]
        r100_set.update(top100_ids)

    r100_list = sorted(list(r100_set))  # optional: sorted for readability

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(r100_list, f, indent=2, ensure_ascii=False)

    print(f"[Saved] R100 set of {len(r100_list)} unique image IDs → {save_path}")


def evaluate_recall_positions(retrieval_path, groundtruth_mapping_path, output_path="recall_position_labels.json"):
    """
    Evaluate recall positions based on true correct image IDs.
    
    Output labels:
    - 0: correct match at position 1
    - 1: correct match in top-10 but not at position 1
    - 10: correct match not in top-10
    - -1: query_id not found in groundtruth mapping
    """
    with open(retrieval_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    with open(groundtruth_mapping_path, "r", encoding="utf-8") as f:
        correct_mapping = json.load(f)

    results = {}

    for query_id, info in retrieval_data.items():
        top10_ids = info.get("retrieved_image_ids", [])
        correct_id = correct_mapping.get(query_id)

        if correct_id is None:
            results[query_id] = -1
        elif correct_id == top10_ids[0]:
            results[query_id] = 0
        elif correct_id in top10_ids:
            results[query_id] = 1
        else:
            results[query_id] = 10

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Recall position labels saved to {output_path}")
    return results

def extract_recall1_correct_mapping(retrieval_path, hand_labels_path, output_path="recall1_correct_mapping.json"):
    """
    Creates a dict {query_id: top1_id} only for queries where Recall@1 is correct (label == 0).
    Saves result as JSON.
    """
    with open(retrieval_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    with open(hand_labels_path, "r", encoding="utf-8") as f:
        human_labels = json.load(f)

    correct_mapping = {}

    for query_id, info in retrieval_data.items():
        top1_list = info.get("retrieved_image_ids", [])
        if not top1_list:
            continue  # skip if no retrievals

        top1_id = top1_list[0]
        if human_labels.get(query_id, 0) == 0:
            correct_mapping[query_id] = top1_id

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(correct_mapping, f, indent=2)

    print(f"✅ Saved {len(correct_mapping)} Recall@1 correct mappings to {output_path}")
    return correct_mapping
    
def get_number_image_top_k(path, topk):
    retrieve = read_json(path)
    sset = set()
    for query in retrieve:
        top_k_images = retrieve[query]["retrieved_image_ids"][:topk]
        sset.update(top_k_images)
    return len(sset)

def add_article_content_query(json_path, database_path):
    query_data = read_json(json_path)
    db_data = read_json(database_path)

    for query in query_data:
        article_id = query["article_id_1"]
        if article_id in db_data:
            query["article_content"] = db_data[article_id]["content"]

    save_json(query_data, json_path)


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def display_ovl(overlaps):
    query_folder = "query_image_private"
    train_folder = "database_images_compressed90_scaled05"

    st.title("Query-Train Image Overlaps")

    for pair in overlaps:
        query_path = os.path.join(query_folder, pair["query_id"] + ".jpg")
        train_path = os.path.join(train_folder, pair["image_id_a"] + ".jpg")

        # Show only if both images exist
        if os.path.exists(query_path) and os.path.exists(train_path):
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(query_path), caption=f"Query: {pair['query_id']}", use_column_width=True)
            with col2:
                st.image(Image.open(train_path), caption=f"Train: {pair['image_id_a']}", use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"Missing image for pair: {pair}")
    return

def check_train_overlap2(query_path, image=True):
    train_data = read_csv("gt_train.csv")      # list of dicts
    query_data = read_csv(query_path)          # list of dicts

    if image:
        key_train = "retrieved_image_id"
        key_query = "image_id_a"
    else:
        key_train = "retrieved_article_id"
        key_query = "article_id_1"

    # Map train IDs to captions
    id_to_caption = {
        item[key_train]: item["caption"]
        for item in train_data
        if key_train in item and "caption" in item
    }

    # Collect overlapping keys
    query_keys = {item[key_query] for item in query_data if key_query in item}
    overlap = set(id_to_caption.keys()) & query_keys

    print(f"Found {len(overlap)} overlapping IDs:")

    # Store query_id for each overlap
    overlapping_pairs = []
    for item in query_data:
        if key_query in item and item[key_query] in overlap:
            overlapping_pairs.append({
                "query_id": item["query_id"],
                key_query: item[key_query]
            })

    for pair in overlapping_pairs:
        print(pair)

    return overlapping_pairs

def check_train_overlap(query_path, image=True):
    train_data = read_csv("gt_train.csv")  # list of dicts
    query_data = read_json("fuse_private_test.csv")     # list of dicts

    if image:
        key_train = "retrieved_image_id"
        key_query = "retrieved_image_id1"
    else:
        key_train = "retrieved_article_id"
        key_query = "article_id_1"

    # Create mapping from ID to caption in train data
    id_to_caption = {item[key_train]: item["caption"] for item in train_data if key_train in item and "caption" in item}

    # Find overlap
    query_keys = {item[key_query] for item in query_data if key_query in item}
    overlap = set(id_to_caption.keys()) & query_keys

    print(f"Found {len(overlap)} overlapping IDs:")
    for item in overlap:
        print(item)

    # Inject caption from train_data into query_data where matched
    for query in query_data:
        q_id = query.get(key_query)
        if q_id in id_to_caption:
            query["generated_caption"] = id_to_caption[q_id]

    save_json(query_data, query_path)

    return overlap, query_data  

def get_stats_len(article_len):
    if 1000 <= article_len <= 2000:
        return 111
    elif article_len > 4000:
        return 110
    else:
        return 112

def add_len_article_result(json_path):
    lsquery = read_json(json_path)
    for query in lsquery:
        content = query["article_content"]
        content = clean_text(content)
        content = normalize_spaces(content)
        query["article_len"] = word_length(content)
    save_json(lsquery, json_path)

def calculate_sste_train(csv_path):
    data = read_csv(csv_path)
    ls_len = []
    for qq in data:
        lenn = word_length(remove_punctuation(qq["caption"]))
        #if lenn >= 95 and lenn <= 130:
        ls_len.append(lenn)

    # Calculate mean
    #mean_len = sum(ls_len) / len(ls_len)
    mean_len = 111

    # Calculate SST
    sste = sum(10 - 10*math.exp(-(x-mean_len)**2/36) for x in ls_len)
    #sst = sum(((x-mean_len)**2) for x in ls_len)
    return sste/len(ls_len), len(ls_len)

def count_caption_length_bins(data, bin_size=10, key="caption"):
    bin_counter = Counter()

    for item in data:
        caption = item.get(key, "")
        word_count = len(remove_punctuation(caption).split())
        bin_index = word_count // bin_size
        bin_counter[bin_index] += 1

    # Convert bin_index to range strings like "0-9", "10-19"
    bin_ranges = {
        f"{i * bin_size}-{(i + 1) * bin_size - 1}": count
        for i, count in bin_counter.items()
    }

    bin_ranges = dict(sorted(bin_ranges.items()))

    save_json(bin_ranges, "val/bin_counter.json")

    return

def generate_test(sample_size=1000):
    dataset = read_json("database.json")
    gt = read_csv("gt_train.csv")

    sample = random.sample(gt, sample_size)

    for item in sample:
        item["query_id"] = item.pop("image_index", None)
        item["gt"] = item.pop("caption", None)
        item["article_id_1"] = item.pop("retrieved_article_id", None)
        item["retrieved_image_id1"] = item.pop("retrieved_image_id", None)
        
        article_data = dataset[item["article_id_1"]]
        item["url"] = article_data["url"]
        item["article_content"] = article_data["content"]
        #item["title"] = article_data["title"]
        item["generated_caption"] = ""

    count_caption_length_bins(sample, 5, "gt")

    save_json(sample, f"val/sample_gt{sample_size}.json")

def merge_key_json(jsonpath1, jsonpath2, mkey):
    data1 = read_json(jsonpath1)
    data2 = read_json(jsonpath2)

    for query2 in data2:
        for query1 in data1:
            if query2["query_id"] == query1["query_id"]:
                query1[mkey] = query2[mkey]
    
    save_json(data1, jsonpath1)

def build_user_qwen_gencap_prompt(query):
    prompt = (
        "Create a detail caption by combining both the image caption and the information of article I provide. "
        "Only return plain text with no formatting:"
        + "\nThe image description:\n" + query["image_raw_caption"] 
        + "\nThe image caption from the article: " + query["web_caption"]
        + "\nThe summary of article:" 
        + "\nTitle: " + query["title"]
        + "\n" + query["summary_article_content"]
    )
    return prompt

def build_user_vlm_gencap_prompt(query):
    format_prompt = f'Article:' \
        '\n"""' \
        f'\nTitle: {query["title"]}' \
        '\nContent:' \
        f'\n{query["article_content"]}' \
        '\n"""' \
        '\nKey Information:' \
        '\n"""' \
        f'\n{query["key_extract"]}' \
        '\n"""' \
        '\nCaption:' \
        f'\n{query["web_caption"]}' \
        '\n"""'
    return format_prompt

def build_user_extract_key_prompt(query):
    format_prompt = f'Article:' \
        '\n"""' \
        f'\nTitle: {query["title"]}' \
        '\nContent:' \
        f'\n{query["article_content"]}' \
        '\n"""' \
        '\nCaption:' \
        f'\n{query["web_caption"]}' \
        '\n"""' 
    return format_prompt

def add_article_title(json_path, db_path):
    query_data = read_json(json_path)
    db_data = read_json(db_path)
    for query in query_data:
        article_id = query["article_id_1"]
        query["title"] = db_data[article_id]["title"]
    save_json(query_data, json_path)

def remove_punctuation_loose(sentence):
    # Punctuations to be removed
    PUNCTUATIONS = ["''", "``", "-LRB-", "-RRB-", "-LCB-", "-RCB-", 
                ".", "?", "!", ",", ":", "--", "...", ";"]
    # Escape each punctuation for regex and sort by length (longer first to avoid partial matches)
    punct_pattern = '|'.join(re.escape(p) for p in sorted(PUNCTUATIONS, key=len, reverse=True))
    # Remove punctuations using regex
    sentence = re.sub(r'\s*(' + punct_pattern + r')\s*', ' ', sentence)
    # Normalize whitespace
    return re.sub(r'\s+', ' ', sentence).strip()

def remove_punctuation(sentence):
    # Punctuations to be removed
    PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", 
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]
    # Escape each punctuation for regex and sort by length (longer first to avoid partial matches)
    punct_pattern = '|'.join(re.escape(p) for p in sorted(PUNCTUATIONS, key=len, reverse=True))
    # Remove punctuations using regex
    sentence = re.sub(r'\s*(' + punct_pattern + r')\s*', ' ', sentence)
    # Normalize whitespace
    return re.sub(r'\s+', ' ', sentence).strip()

def sort_list_of_dicts(data, sort_key, reverse=False):
    return sorted(data, key=lambda x: x.get(sort_key), reverse=reverse)

def set_anormaly_query(query_data, thr):
    o_set = set()
    maxx, minn, summ = 0, 1000, 0
    maxx_, minn_, summ_ = 0, 1000, 0
    for query in query_data:
        gencap = query["image_raw_caption"]
        gencap = clean_text(gencap)
        gencap = normalize_spaces(gencap)
        wlen = word_length(gencap)

        gencap_ = query["generated_caption"]
        gencap_ = clean_text(gencap_)
        gencap_ = normalize_spaces(gencap_)
        wlen_ = word_length(gencap_)
        if wlen <= thr:
            o_set.add(query["retrieved_image_id1"])
            maxx = max(maxx, wlen)
            minn = min(minn, wlen)
            summ += wlen

            maxx_ = max(maxx_, wlen_)
            minn_ = min(minn_, wlen_)
            summ_ += wlen_
    print(f"Searching find {len(o_set)}")
    print(f"Image caption: Max {maxx}, Min {minn}, Mean {summ/len(o_set)}")
    print(f"Generated caption: Max {maxx_}, Min {minn_}, Mean {summ_/len(o_set)}")
    return o_set

def copy_anomaly_images(out_json_path, image_folder, anomaly_folder):
    # Ensure anomaly folder exists
    os.makedirs(anomaly_folder, exist_ok=True)

    # Load output JSON file
    with open(out_json_path, "r") as f:
        data = json.load(f)

    count = 0
    for image_id, _ in data:
        # Construct full paths
        src_path = os.path.join(image_folder, image_id+".jpg")
        dst_path = os.path.join(anomaly_folder, image_id+".jpg")

        # Copy image if it exists
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            count += 1
        else:
            print(f"Image not found: {src_path}")
    print(f"Copied {count} images to '{anomaly_folder}'")

def search_short_querystr(json_path, key, thr):
    data = read_json(json_path) 
    outp = []
    maxx, minn, summ = 0, 1000, 0
    for query in data:
        gencap = query[key]
        gencap = clean_text(gencap)
        gencap = normalize_spaces(gencap)
        wlen = word_length(gencap)
        if wlen <= thr:
            outp.append((query["retrieved_image_id1"], wlen))
            maxx = max(maxx, wlen)
            minn = min(minn, wlen)
            summ += wlen
    save_json(outp, "mining/out.json")
    print(f"Searching find {len(outp)}")
    print(f"Max {maxx}, Min {minn}, Mean {summ/len(outp)}")
    return outp

def search_image_train(train_csv_path, thr):
    data = read_csv(train_csv_path)
    outp = []
    for line in data:
        gencap = line["caption"]
        gencap = clean_text(gencap)
        gencap = normalize_spaces(gencap)
        wlen = word_length(gencap)
        if wlen <= thr:
            outp.append((line["retrieved_image_id"],wlen,line["caption"]))
    save_json(outp, "mining/out.json")
    print(f"Searching find {len(outp)}")
    return wlen

def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def build_y_x_word_train_data(train_csv_path, database_path, nm_train=1, nm_atc=1):
    x_article_len = []
    y_gencap_len = []
    train_data = read_csv(train_csv_path)
    database_data = read_json(database_path)
    
    for idx, line in enumerate(train_data):
        gencap = line["caption"]
        gencap = clean_text(gencap)
        gencap = normalize_spaces(gencap)
        x_article_len.append(word_length(gencap) // nm_train)
        
        article_id = line["retrieved_article_id"]
        article_content = database_data[article_id]["content"]
        article_content = clean_text(article_content)
        article_content = normalize_spaces(article_content)
        y_gencap_len.append(word_length(article_content) // nm_atc)

    return y_gencap_len, x_article_len

def delete_key_query_content(json_path, keyy, save_path, 
                             min_thr, just_count=False, print_=False):
    count = 0
    data = read_json(json_path)
    for query in data:
        if word_length(query[keyy]) < min_thr:
            if just_count == False:
                query[keyy] = ''
            if print_ == True:
                print(query["article_id_1"])
                #print(query[keyy])
            count += 1
    save_json(data, save_path)
    print(f"Number of delete {count}")
    return count

def zip_folder(folder_path="submission", output_zip="submission.zip"):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)  # Preserve folder structure
                zipf.write(abs_path, rel_path)
    print(f"Folder '{folder_path}' zipped to '{output_zip}'")

def format_lines_in_file(file_path="submission/submission.csv"):
    try:
        with open(file_path, 'r') as f_in:
            lines = f_in.readlines()

        formatted_lines = []
        for idx, line in enumerate(lines):
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                formatted_lines.append("")
                continue

            last_comma_index = line.rfind(',')

            if last_comma_index != -1:
                # Insert " after the last comma
                modified_line = (
                    line[:last_comma_index + 1]
                    + '"'
                    + line[last_comma_index + 1:]
                )
            else:
                modified_line = line

            modified_line += '"'
            formatted_lines.append(modified_line)

        with open(file_path, 'w') as f_out:
            for formatted_line in formatted_lines:
                f_out.write(formatted_line + '\n')

        print(f"File '{file_path}' has been successfully formatted.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def clean_text(text: str) -> str:
    # Keep only letters and spaces
    return re.sub(r'[^a-zA-Z\s]', '', text).strip().replace("\n","")

def percentOOV_ngram(cap: str, ngram: List[Union[int, str, List[str], float]]) -> float:
    # Percent out of vocab of a sentence
    ans = 0
    for word in ngram[2]:
        if word not in cap:
            ans += 1
    return ans * 1.0 / ngram[0]

def chunking_words(cap: str, word_len:int):
    return " ".join(cap.split()[:word_len])

def word_length(cap):
    return len(cap.split(" "))

def image_to_base64_path(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string 

def merge_image_summary_cap(img_json_path, sum_json_path):
    img_data= read_json(img_json_path)
    sum_data = read_json(sum_json_path)
    for img_query in img_data:
        for new_query in sum_data:
            if img_query["query_id"] == new_query["query_id"]:
                new_query["image_raw_caption"] = img_query["generated_caption"]
                new_query["retrieved_image_id1"] = img_query["retrieved_image_id1"]
    save_json(sum_data, sum_json_path)

def delete_key_content(json_path, keyy, save_path):
    data = read_json(json_path)
    for query in data:
        query[keyy] = ''
    save_json(data, save_path)

def remove_key_format_query(json_path, keyy):
    data = read_json(json_path)
    for query in data:
        query.pop(keyy, None)
    save_json(data, json_path)

def add_key_format_query(json_path, key):
    data = read_json(json_path)
    for query in data:
        query[key] = ""
    save_json(data, json_path)

def merge_query_previous(json_path1, json_path2):
    first_data = read_json(json_path1)
    second_data = read_json(json_path2)
    for cap_query in second_data:
        for new_query in first_data:
            if cap_query["query_id"] == new_query["query_id"] \
                and cap_query["article_id_1"] == new_query["article_id_1"]:
                new_query["summary_article_content"] = cap_query["summary_article_content"]
    save_json(first_data, json_path1)

def merge_query(json_path, max_id):
    first_data = read_json("format_query1.json")
    ls_data=[]
    for i in range(1,max_id+1):
        data=read_json(f"format_query{i}.json")
        ls_data.append(data)
    for format_query in ls_data:
        for idx, query in enumerate(format_query):
            if first_data[idx]["summary_article_content"] == "" \
                and query["summary_article_content"] != "":
                first_data[idx]["summary_article_content"] = query["summary_article_content"]
    save_json(first_data, json_path)

def create_query_json(query_csv, database_json, name_json):
    ls_query = []
    for row_query in query_csv:
        new_query = {}
        new_query["query_id"] = row_query['query_id']
        new_query["article_id_1"] = row_query["article_id_1"]
        new_query["article_content"] = database_json[row_query["article_id_1"]]['content']
        new_query["summary_article_content"] = ""
        new_query["generated_caption"] = ""
        ls_query.append(new_query)
    save_json(ls_query, name_json)
    return ls_query

def save_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_csv(csv_path):
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def write_csv(data, path):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)