from utils.util import *
from utils.post_module import *
from utils.converter import *
from utils.truncate_captions import *

# INPUT
query_path = "json_query/input.json"
csv_path = "submission_98.csv"

# CONFIG
max_words = 105
cutting_idx = 104

# OUTPUT --> submission.zip
tmp_save_path = "refinement.json"
output_file="json_query/metadata_truncated.json"

stats = sematicNorm(
        query_path, output_file,
        max_word=105, expected_length=104,
        verbose=True, dynamic=False
    )

entityEnricker(
    output_file, max_words, cutting_idx,
    tmp_save_path, "truncated_caption", 0, False)

fquery_to_queryid_cp()

join_caps_to_csv(csv_path)

zip_folder()

