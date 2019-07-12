# THINGS WE SHOULD DO:
# 1) Figure out which params are best to consider -- do via cross validation and pick the set with best performance
# 2)

import pandas as pd
import requests
import json

BASE_URL = "http://127.0.0.1:5000/train"
RESPONSE_DATA = "./data/expert_grader_valid_100.csv"

PARSER_FEATURE_LIST = ["bad_word_count",
                       "domain_word_count",
                       "innovation_word_count",
                       "common_word_count",
                       "stem_word_count",
                       "option_word_count"]

# Set classifier features to False or True to exclude/include them
# Set parser params to False or True to exclude/include them
app_params = {
    "bad_word_count": True,
    "domain_word_count": False,
    "innovation_word_count": False,
    "common_word_count": True,
    "stem_word_count": True,
    "option_word_count": True,
    "remove_stopwords": True,
    "tag_numeric": True,
    "spelling_correction": True,
    "remove_nonwords": True,
}

# Load the data, convert to json for transport
df = pd.read_csv(RESPONSE_DATA)
df = df.rename(columns = {"valid": "valid_label"})
df_json = df.to_json()

# Call the train route, get response
# Grab both the parsed out dataframe and the coefficients (+ intercept)
r = requests.get(BASE_URL, params=app_params, json={"response_df": df_json})
out = r.json()
features_to_consider = [v for v in PARSER_FEATURE_LIST if app_params[v]]
columns = ["response"] + features_to_consider + ["valid_label"]
df_out = pd.read_json(out.pop('output_df',  None)).sort_index()
df_out = df_out[columns]

# Print out the coefficients/intercept
for k in out.keys():
    print("{}:{}".format(k, out[k]))


