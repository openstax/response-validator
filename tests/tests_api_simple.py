# tests_api.py
# Author: drew
# Measures the tradeoffs in timing/accuracy for different configurations of the response parser and different levels of complexity in how we count vocabulary
# Creates two dataframes:
# a) api_simuilation_results.csv: A table of 2**5 rows with average timing and accuracy data (one row for each possible configuration)
# b) api_simulation_delta.csv: A table showing the average gain in accuracy/timing for each of the five possible parameters

import pandas as pd
import numpy as np
import time
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import requests

# Users params -- where to get the response data and how much of it to sample for testing
base_url = "http://127.0.0.1:5000/validate"  # Local Path
# base_url = 'https://protected-earth-88152.herokuapp.com/validate' #Heroku path
response_path = "~/Box Sync/Research/Data/openform_app/response_data.csv"
n_samp = 100

# Read the data, set up the default LogisticRegression model
df_full = pd.read_csv(response_path)
df_full = df_full[df_full["subject_name"] == "Biology"]
df = df_full.sample(n=n_samp, random_state=42)
lr = LogisticRegression()

# Simple helper function to process the result of the api call into something nice for a pandas dataframe
# Simple helper function to process the result of the api call into something nice for a pandas dataframe
def do_api_time_call(response, uid, stops, nums, spell, nonwords, use_uid):
    if not use_uid:
        uid = None
    params = {
        "response": response,
        "uid": uid,
        "remove_stopwords": stops,
        "tag_numeric": nums,
        "spelling_correction": spell,
        "remove_nonwords": nonwords,
    }
    r = requests.get(base_url, params=params)
    D = r.json()
    computation_time = str(D["computation_time"])
    validity_label = D["valid"]
    return ",".join([computation_time, str(validity_label)])


# Iterate through all parser/vocab combinations and get average timing estimates per response
# Then do a 5-fold cross validation to estimate accuracy
print("Starting the test")


stops = True
nums = False
spell = True
nonwords = True
use_uid = True

# Make a local copy of the dataframe
dft = df.copy()

# Compute the actual features and do the timing computation (normalized per response)
now = time.time()
dft["time_valid"] = dft.apply(
    lambda x: do_api_time_call(
        x.free_response, x.uid, stops, nums, spell, nonwords, use_uid
    ),
    axis=1,
)
elapsed_time_total = time.time() - now
elapsed_time = elapsed_time_total / n_samp

dft["computation_time"] = (
    dft["time_valid"].apply(lambda x: x.split(",")[0]).astype(float)
)
dft["valid_prediction"] = dft["time_valid"].apply(lambda x: x.split(",")[1])
dft["valid_prediction"] = dft["valid_prediction"].map(
    {"True": "Valid", "False": "Invalid"}
)

computation_time = dft["computation_time"].mean()
dft["pred_correct"] = dft["valid_prediction"] == dft["junk"]
print("Avg. Computation Time: " + str(computation_time))
print("Avg. End-To-End Time: " + str(elapsed_time))
print("Computation Time Percentage of Total: " + str(computation_time / elapsed_time))
print("Prediction Accuracy: " + str(dft["pred_correct"].mean()))
