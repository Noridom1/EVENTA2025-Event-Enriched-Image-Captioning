import json
import spacy
from statistics import mean
from tqdm import tqdm
import sys
from utils.util import *
from spacy.tokens import Doc

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", 
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]

def rebuild_caption(tokens):
    caption = ""
    for i, token in enumerate(tokens):
        if i > 0:
            caption += " "
        caption += token
    return caption


def is_name_entity(word, entity_names):
    word = remove_punctuation_loose(word)
    if word in entity_names:
        return True
    return False

def sematicNorm(
    input_file: str,
    output_file: str = "metadata_truncated.json",
    expected_length: int = 111,
    max_word: int = 120,
    verbose: bool = True,
    dynamic: bool = False,
):
    nlp = spacy.load("en_core_web_sm")

    def truncate_caption(text, max_word, expected_length):
        text = normalize_spaces(text.replace("\n",""))
        if word_length(remove_punctuation(text)) <= max_word:
            return text 
        
        word_count = word_length(text)
        doc = nlp(text)
        #entity_names = [ent.text for ent in doc.ents]
        entity_names = set(word for ent in doc.ents for word in ent.text.split())
        
        ls_word = text.split(" ")
        entity_mask = [False] * word_count
        for idx, wordi in enumerate(ls_word):
            entity_mask[idx] = is_name_entity(wordi, entity_names)
        word_entity_pairs = list(zip(ls_word, entity_mask))

        # Truncate non-entity words from the end
        truncated = []
        count = 0
        ngram_count = word_length(remove_punctuation(text))
        for word, is_entity in reversed(word_entity_pairs):
            if not is_entity: # Not combine
                if ngram_count - count > expected_length:
                    count += 1
                    continue
            truncated.append((word, is_entity))

        truncated = list(reversed(truncated))
        final_tokens = [tok for tok, _ in truncated]

        caption = rebuild_caption(final_tokens).replace(",.", ".").replace(".,", ".")

        if caption[-1].isalpha():
            caption += '.'
        else:
            caption = caption[:-1] + "."

        return caption

    data = read_json(input_file)

    lengths = []
    for result in tqdm(data, desc="Processing captions"):
        #caption = result.get("generated_caption", "")
        caption = result.get("generated_caption", "")
        max_word_ = max_word
        expected_length_ = expected_length
        if dynamic:
            max_word_ = expected_length_ = get_stats_len(result["article_len"])
        truncated = truncate_caption(caption, max_word_, expected_length_)
        result["truncated_caption"] = truncated
        lengths.append(word_length(remove_punctuation(truncated)))

    if verbose:
        print("Truncated Caption Statistics:")
        print(f"Mean Length: {mean(lengths):.2f} words")
        print(f"Min Length: {min(lengths)} words")
        print(f"Max Length: {max(lengths)} words")

    save_json(data, output_file)