from utils.util import *
from utils.extractor import *

def concat_ngram(sort_df, cap, max_words, percent00V:float):
    if word_length(cap) >= max_words:
        return cap
    if len(cap) == 1:
        cap = ""

    for ngram in sort_df:
        if word_length(cap) >= max_words:
            break
        if percentOOV_ngram(cap, ngram) >= percent00V:
            cap += " " + ngram[1]
    
    return cap

def concat_word(cap, query, cutting, word_type):
    if word_length(remove_punctuation(cap)) >= cutting:
        return cap
    
    ls_key = ["web_caption", "summary_article_content", "title", "article_content"]

    
    for key in ls_key:
        ls_entity_summary = TokenExtractor.extract_word_type(
            text=query[key],word_type=word_type
        )
        for entity in ls_entity_summary:
            if word_length(remove_punctuation(cap)) >= cutting:
                return cap
            if entity not in remove_punctuation(cap):
                cap += " " + entity.replace("<alt>","").replace("<alt","")
    return cap

def chunking_anomaly(set_anorm, query_data, max_w, cutt):
    for query in query_data:
        cap = query["generated_caption"]
        if query["retrieved_image_id1"] in set_anorm:
            if word_length(cap) > max_w:
                cap = chunking_words(cap, cutt)
        query["generated_caption"] = cap

def entityEnricker(
    query_path, 
    word_thr, 
    cutting_idx, 
    save_path, 
    usekey, 
    df_offset=0, 
    dynamic=False,
    ):
    #sort_df = read_json("ngram_df/sort_df.json")
    query_data = read_json(query_path)

    sum_word_len = 0
    max_word_len = 0
    min_word_len = 1000
    ls_word_lens = []

    for idx ,query in enumerate(tqdm(query_data, desc="Concate and chunking words")):
        cap = (query[usekey])
        word_thr_ = word_thr
        cutting_idx_ = cutting_idx

        if dynamic:
            word_thr_ = cutting_idx_ = get_stats_len(query["article_len"])

        cap = concat_word(cap, query, cutting_idx_, "name_entity")
        cap = concat_word(cap, query, cutting_idx_, "noun_phrases")
        w_len = word_length(remove_punctuation(cap))
        
        if w_len > word_thr_:
            cap = chunking_words(cap, cutting_idx_)
            w_len = cutting_idx_

        sum_word_len += w_len
        max_word_len = max(max_word_len, w_len)
        min_word_len = min(min_word_len, w_len)
        ls_word_lens.append(w_len)

        query["generated_caption"] = cap
        query["rf_len"] = word_length(cap)
    
    query_data = sort_list_of_dicts(query_data, "rf_len", True)

    for idx, query in enumerate(query_data):
        query["id"] = idx+1


    save_json(query_data, f"refinement_json/{save_path}")
    print(f"Mean word length: {sum_word_len/len(query_data):.6f}")
    print(f"Max word length: {max_word_len}")
    print(f"Min word length: {min_word_len}")
    return

def partial_post_process(query_path, word_thr, cutting_idx, save_path, usekey, start_id, end_id):
    sort_df = read_json("ngram_df/sort_df.json")
    query_data = read_json(query_path)

    sum_word_len = 0
    max_word_len = 0
    min_word_len = 1000

    # Old gencap
    for idx, query in enumerate(query_data):
        if idx >= start_id-1 and idx <= end_id-1:
            continue
        cap = remove_punctuation(query_data[idx][usekey])
        cap = normalize_spaces(cap)
        cutting = query["rf_len"]
        cap = chunking_words(cap, cutting)
        query_data[idx]["generated_caption"] = cap

    # Refinement in range
    for idx in range(start_id-1, end_id):
        cap = remove_punctuation(query_data[idx][usekey])
        cap = normalize_spaces(cap)

        #cap = concate_ngram(sort_df, cap, word_thr, 0.6)

        if word_length(cap) > word_thr:
            cap = chunking_words(cap, cutting_idx)

        sum_word_len += word_length(cap)
        max_word_len = max(max_word_len, word_length(cap))
        min_word_len = min(min_word_len, word_length(cap))

        query_data[idx]["generated_caption"] = cap
        query_data[idx]["rf_len"] = word_length(cap)

    # set_arnom = set_anormaly_query(query_data, 35)
    # chunking_anomaly(set_arnom, query_data, 50, 50)
    
    #query_data = sort_list_of_dicts(query_data, "rf_len", True)


    save_json(query_data, f"refinement_json/{save_path}")
    print(f"Mean word length: {sum_word_len/(end_id-start_id+1):.6f}")
    print(f"Max word length: {max_word_len}")
    print(f"Min word length: {min_word_len}")
    return 