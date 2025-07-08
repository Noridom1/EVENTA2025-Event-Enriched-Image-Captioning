from utils.util import *

def fquery_to_queryid_cp(json_path="refinement_json/refinement.json"
                         , target="tmp/tmp.json"):
    data = read_json(json_path)
    dicc = {}
    for query in data:
        dicc[query["query_id"]] = query["generated_caption"].replace("\n","")
    save_json(dicc,target)

def write_csv_dict(path, data, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def join_caps_to_csv(csv_path, queryid_cap_path="tmp/tmp.json"
                     , target="submission/submission.csv"):
    queryid_cap = read_json(queryid_cap_path)
    csv_data = read_csv(csv_path)

    for row in csv_data:
        query_id = row["query_id"]
        if query_id in queryid_cap:
            row["generated_caption"] = queryid_cap[query_id].strip()

    write_csv_dict(target, csv_data, fieldnames=csv_data[0].keys())


def join_caps_to_csv(csv_path, queryid_cap_path="tmp/tmp.json"
                     , target="submission/submission.csv"):
    queryid_cap = read_json(queryid_cap_path)
    csv_data = read_csv(csv_path)

    for row in csv_data:
        query_id = row["query_id"]
        if query_id in queryid_cap:
            row["generated_caption"] = queryid_cap[query_id].strip()

    write_csv_dict(target, csv_data, fieldnames=csv_data[0].keys())

def join_caps_to_csv_meow(csv_path, queryid_cap_path="tmp/tmp.json", target="submission/submission.csv"):
    queryid_cap = read_json(queryid_cap_path)
    csv_data = read_csv(csv_path)

    for idx, row in enumerate(csv_data):
        query_id = row["query_id"]
        if idx >= 1000:
            meow = "meow"
            row["generated_caption"] = f'"{meow}"'
        elif query_id in queryid_cap:
            row["generated_caption"] = queryid_cap[query_id].strip()

    write_csv_dict(target, csv_data, fieldnames=csv_data[0].keys())