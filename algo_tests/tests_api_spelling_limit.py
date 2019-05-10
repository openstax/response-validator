# tests_api.py
# Author: drew
# Measures the tradeoffs in timing/accuracy for different configurations of the response parser

import pandas as pd
import time
from itertools import product
import matplotlib as mpl

mpl.use("TkAgg")
from plotnine import *
import requests

# Users params -- where to get the response data and how much of it to sample for testing
BASE_URL = "http://127.0.0.1:5000/validate"
# Local Path 'https://protected-earth-88152.herokuapp.com/validate' #Heroku path
STOPS = [True]
NUMS = ["auto"]
SPELL = [True, "auto"]
NONWORDS = [True]
USE_UID = [True]
SPELLING_LIMIT = range(0, 10)
DATAPATHS = ["./data/expert_grader_valid_100.csv", "./data/alicia_valid.csv"]
COLUMNS = [
    "bad_word_count",
    "common_word_count",
    "computation_time",
    "domain_word_count",
    "inner_product",
    "innovation_word_count",
    "num_spelling_correction",
    "processed_response",
    "remove_nonwords",
    "remove_stopwords",
    "response",
    "spelling_correction",
    "spelling_correction_used",
    "tag_numeric",
    "tag_numeric_input",
    "uid_found",
    "uid_used",
    "valid_result",
]


# Simple helper function to process the result of the api call into something nice for a pandas dataframe
def do_api_time_call(
    response, uid, stops, nums, spell, nonwords, use_uid, spelling_limit
):
    if not use_uid:
        uid = None
    params = {
        "response": response,
        "uid": uid,
        "remove_stopwords": stops,
        "tag_numeric": nums,
        "spelling_correction": spell,
        "remove_nonwords": nonwords,
        "spell_correction_max": spelling_limit,
    }
    r = requests.get(BASE_URL, params=params)
    return_dictionary = r.json()
    strings = [str(return_dictionary[k]) for k in return_dictionary.keys()]
    return "xxx".join(strings)


# Iterate through all parser/vocab combinations and get average timing estimates per response
# Then do a 5-fold cross validation to estimate accuracy
print("Starting the test")

df_results = pd.DataFrame()

for datapath, stops, nums, spell, nonwords, use_uid, spelling_limit in product(
    DATAPATHS, STOPS, NUMS, SPELL, NONWORDS, USE_UID, SPELLING_LIMIT
):
    # Load the data
    dft = pd.read_csv(datapath)
    dft["data"] = datapath
    n_samp = dft.shape[0]

    # Compute the actual features and do the timing computation (normalized per response)
    now = time.time()
    dft["result"] = dft.apply(
        lambda x: do_api_time_call(
            x.free_response,
            x.uid,
            stops,
            nums,
            spell,
            nonwords,
            use_uid,
            spelling_limit,
        ),
        axis=1,
    )
    elapsed_time_total = time.time() - now
    elapsed_time = elapsed_time_total / n_samp

    dft_results = dft.result.str.split("xxx", expand=True)
    dft_results.columns = COLUMNS
    dft_results = pd.concat([dft, dft_results], axis=1)
    dft_results["computation_time"] = dft_results["computation_time"].astype(float)
    dft_results["spelling_limit"] = spelling_limit
    dft_results["pred_correct"] = (
        dft_results["valid_result"].map({"True": True, "False": False})
        == dft_results["valid"]
    )
    df_results = df_results.append(dft_results)

df_results = df_results.reset_index()

# Plot computation time (mean, min, max) for the various cases
df_results["short_name"] = df_results["data"].apply(
    lambda x: x.split("/")[-1].split("_")[0] + "_data"
)
df_results["spelling_str"] = df_results["spelling_correction"].apply(
    lambda x: "Spell=" + x
)
df_results["number_str"] = df_results["tag_numeric_input"].apply(lambda x: "Num=" + x)
df_results["spelling_limit_str"] = df_results["spelling_limit"].apply(lambda x: str(x))
plot_time = (
    ggplot(df_results, aes("spelling_limit_str", "1000*computation_time"))
    + geom_violin()
    + facet_grid("short_name~number_str")
    + xlab("Spelling Correction")
    + ylab("Computation Time (msec)")
)

print(plot_time)
