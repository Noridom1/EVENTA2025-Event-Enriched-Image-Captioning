import os
import json
import time
import random
import urllib.robotparser
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin
from utils import *
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_images_from_cnn(url, output_dir, headers):
    image_caption_pairs = {}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to access {url} - Status Code {response.status_code}")
            return image_caption_pairs, f"❌ Failed to access {url} - Status Code {response.status_code}"

        soup = BeautifulSoup(response.content, "html.parser")
        main_content = soup.find("main", class_="article__main")
        
        if not main_content:
            print(f"⚠️ No <main class='article__main'> found in {url}")
            # return image_caption_pairs, f"⚠️ No <main class='article__main'> found in {url}"
            main_content = soup
            image_blocks = main_content.find_all('div', class_='media')
            img_count = 0
            for image_block in image_blocks:
                img = image_block.find("img", class_="media__image")
                caption_div = image_block.find("div", class_="media__caption")
                
                if img:
                    # Get full-size or available image source
                    img_url = img.get("data-src-large") or img.get("src") or img.get("data-src")
                    if img_url and img_url.startswith("//"):
                        img_url = "https:" + img_url

                    # Get caption text
                    caption = ""
                    if caption_div:
                        raw = caption_div.find("div", class_="element-raw")
                        if raw:
                            caption = raw.get_text(strip=True)

                    # Save image
                    img_ext = os.path.splitext(img_url.split('?')[0])[-1]
                    filename = f"img_{img_count}{img_ext}"
                    save_path = os.path.join(output_dir, filename)

                    try:
                        time.sleep(random.uniform(0.01, 0.1))
                        img_response = requests.get(img_url, headers=headers)
                        if img_response.status_code == 200:
                            with open(save_path, 'wb') as f:
                                f.write(img_response.content)
                            image_caption_pairs[filename] = caption
                            img_count += 1
                        else:
                            print(f"⚠️ Failed to download image: {img_url}")
                    except Exception as e:
                        print(f"❌ Error downloading image: {img_url} - {e}")
                        continue

        else:
            image_blocks = main_content.find_all('div', class_='image')
            img_count = 0
            for image_block in image_blocks:
                if image_block.find_parent('a') or image_block.find_parent(class_='video-resource__image'):
                    continue

                # Extract image URL
                img_tag = image_block.find('img')
                if not img_tag:
                    continue

                img_url = img_tag.get('src')
                if not img_url:
                    continue

                # Extract caption
                caption_tag = image_block.find('div', class_='image__caption')
                caption = caption_tag.get_text(separator=' ', strip=True) if caption_tag else ""

                # Fallback: use alt text if no caption found
                alt_text = img_tag.get('alt', '').strip()
                if alt_text:
                    caption = caption + '<alt>' + alt_text

                # Save image
                img_ext = os.path.splitext(img_url.split('?')[0])[-1]
                filename = f"img_{img_count}{img_ext}"
                save_path = os.path.join(output_dir, filename)

                try:
                    time.sleep(random.uniform(0.01, 0.1))
                    img_response = requests.get(img_url, headers=headers)
                    if img_response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(img_response.content)
                        image_caption_pairs[filename] = caption
                        img_count += 1
                    else:
                        print(f"⚠️ Failed to download image: {img_url}")
                except Exception as e:
                    print(f"❌ Error downloading image: {img_url} - {e}")
                    continue

        # print("✅ Successfully scraped images from CNN article")
        return image_caption_pairs, "✅ Successfully scraped images from CNN article"

    except Exception as e:
        print(f"❌ Failed to process {url} due to {e}")
        return image_caption_pairs, f"❌ Failed to process {url} due to {e}"


def get_images_from_guardian(url, output_dir, headers):
    image_caption_pairs = {}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return image_caption_pairs, f"❌ Failed to fetch {url}: {e}"

    soup = BeautifulSoup(response.content, "html.parser")
    main_content = soup.find("main")

    if not main_content:
        print("⚠️ <main> content not found.")
        return image_caption_pairs, "⚠️ <main> content not found."

    figures = main_content.find_all("figure")
    image_counter = 1

    for figure in figures:
        # Shared figcaption if available
        figcaption = figure.find("figcaption")
        shared_caption = figcaption.get_text(strip=True) if figcaption else ''

        # Find all img tags inside this figure
        img_tags = figure.find_all("img")
        for img_tag in img_tags:
            img_src = img_tag.get("src")
            if not img_src:
                continue

            img_url = urljoin(url, img_src)  # Keep query params for auth
            img_ext = os.path.splitext(img_url.split("?")[0])[1] or ".jpg"
            img_filename = f"image_{image_counter}{img_ext}"
            img_path = os.path.join(output_dir, img_filename)

            try:
                time.sleep(random.uniform(0.01, 0.1))
                img_response = requests.get(img_url, headers=headers)
                img_response.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(img_response.content)
            except requests.RequestException as e:
                print(f"⚠️ Failed to download {img_url}: {e}")
                continue

            # Determine caption
            alt_text = img_tag.get("alt", "").strip()
            caption = shared_caption + ('<alt>' + alt_text if alt_text else "")

            image_caption_pairs[img_filename] = caption
            image_counter += 1

    # print(f"✅ Completed extraction from Guardian: {len(image_caption_pairs)} images found.")
    return image_caption_pairs, f"✅ Completed extraction from Guardian: {len(image_caption_pairs)} images found."


def process_query(results, entry, output_dir, headers, user_agent, robot_parser_base):
    image_caption_pairs = {}
    query_id = entry.get("query_id")
    if query_id in results and results[query_id] != {}:
                return query_id, results[query_id], f'[Finished] Request {query_id}\n'

    article_id = entry.get("article_id_1")
    url = entry.get("url")
    logging(os.path.join(output_dir, 'log.txt'), f'Request {query_id}\n')

    query_output_dir = os.path.join(output_dir, str(query_id))
    Path(query_output_dir).mkdir(parents=True, exist_ok=True)

    if not url:
        return query_id, {}, f"⚠️ No URL found for article_id {article_id}\n"

    try:
        robots_url = f"{url.split('/')[0]}//{url.split('/')[2]}/robots.txt"
        robot_parser_base.set_url(robots_url)
        robot_parser_base.read()
        if not robot_parser_base.can_fetch(user_agent, url):
            return query_id, {}, f"❌ BLOCKED by robots.txt - {url}\n"
    except Exception as e:
        return query_id, {}, f"⚠️ Could not read robots.txt for {url} due to {e}\n"

    if "cnn.com" in url:
        image_caption_pairs, log_message = get_images_from_cnn(url, query_output_dir, headers)
    elif "theguardian.com" in url:
        image_caption_pairs, log_message = get_images_from_guardian(url, query_output_dir, headers)
    else:
        return query_id, {}, f"⚠️ Unsupported domain - {url}\n"

    return query_id, image_caption_pairs, f"✅ {query_id}: {len(image_caption_pairs)} images; {url}: {log_message}\n"


def fetch_all_images(metadata_path, output_dir, failed_queries=None, run_time=1, user_agent="MyNewsBot", progress_step=20, num_workers=50):
    headers = {"User-Agent": user_agent}
    robot_parser = urllib.robotparser.RobotFileParser()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(output_dir, 'log.txt')
    result_path = os.path.join(output_dir, f'image_caption_pairs.json')

    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        metadata = [md for md in metadata if md['web_caption'] == ""]
        
    if failed_queries:
        metadata = [q for q in metadata if q['query_id'] in failed_queries]

    total = len(metadata)
    print(f"Total number of queries: {total}")

    with ThreadPoolExecutor(num_workers) as executor:
        futures = {
            executor.submit(process_query, results, entry, output_dir, headers, user_agent, robot_parser): entry
            for entry in metadata
        }

        for i, future in enumerate(tqdm(as_completed(futures), total=total)):
            query_id, image_caption_pairs, log_message = future.result()
            
            if results.get(query_id, {}) == {}:
                results[query_id] = image_caption_pairs
                
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_message)

            if (i + 1) % progress_step == 0 or i == total - 1:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

    return results


# Example usage:
metadata_path = '../CaptionEnriching/files/metadata_final.json'
output_dir = 'images_final'
# failed_query_ids = get_failed_query_ids('ImagesFromWeb/image_caption_pairs.json')
# print(failed_query_ids)
results = fetch_all_images(metadata_path, output_dir, failed_queries=None, run_time=1, num_workers=100)
print(len(results))
# with open(f'{output_dir}/image_caption_pairs_2.json', 'w') as f:
#     json.dump(results, f, indent=4)

