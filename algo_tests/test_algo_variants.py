# Quick script to test out some algorithm variants using some test data
# To run me: Be in the directory above this script and do python3 -m algo_tests.test_algo_variants

from validator import app
from urllib.parse import urlencode
from collections import OrderedDict
import pandas as pd
from validator import validate_api

import warnings

warnings.filterwarnings("ignore")


FEATURES_OLD = OrderedDict(
    {
        "stem_word_count": 0,
        "option_word_count": 0,
        "innovation_word_count": 2.2,
        "domain_word_count": 2.5,
        "bad_word_count": -3,
        "common_word_count": 0.7,
        "intercept": 0,
        "lazy_math_mode": False,
    }
)

FEATURES_NEW = OrderedDict(
    {
        "stem_word_count": 2.2,
        "option_word_count": 2.2,
        "innovation_word_count": 0,
        "domain_word_count": 0,
        "bad_word_count": -3,
        "common_word_count": 0.7,
        "intercept": 0,
        "lazy_math_mode": True,
    }
)

# Configure the app
testapp = app.create_app(test_config={"TESTING": True})
client = testapp.test_client()

parser = validate_api.parser

# Load the test data and append together
df1 = pd.read_csv("algo_tests/data/valid_test_1000.csv")
df2 = pd.read_csv("algo_tests/data/invalid_test_1000.csv")
df = df1.append(df2).reset_index()

# The idea here is that we will be able to pass in the route along with any parameters and get the validity label out
def do_api_call(data, route, algo_params):
    params = {"response": data.response, "uid": data.uid}
    params.update(algo_params)
    resp = client.get(route, query_string=urlencode(params))
    return resp


def get_validity(data, route, algo_params):
    resp = do_api_call(data, route, algo_params)
    valid = resp.json["valid"]
    return valid


def do_training(df, train_params):
    params = {}
    params.update(train_params)
    df_train = (
        df[["response", "uid", "valid_truth"]]
        .rename(
            columns={
                "response": "free_response",
                "uid": "uid",
                "valid_truth": "valid_label",
            }
        )
        .to_json()
    )
    resp = client.get(
        "/train", query_string=urlencode(params), json={"response_df": df_train}
    )
    resp_dict = resp.json
    new_params = {k: resp_dict[k] for k in train_params.keys() & resp_dict.keys()}
    return new_params


print("Starting!")

### Do a quick look at performance before and after the major algo changes
df["old"] = df.apply(lambda x: get_validity(x, "/validate", FEATURES_OLD), axis=1)
df["new"] = df.apply(lambda x: get_validity(x, "/validate", FEATURES_NEW), axis=1)

### Look at training -- do a global and per subject train
FEATURES_TRAIN = FEATURES_NEW.copy()
FEATURES_TRAIN["intercept"] = 1
df_bio = df[df["subject"] == "Biology 2e"]
df_phy = df[df["subject"] == "College Physics with Courseware"]
df_soc = df[df["subject"] == "Introduction to Sociology 2e"]
params_bio = do_training(df_bio, FEATURES_TRAIN)
params_phy = do_training(df_phy, FEATURES_TRAIN)
params_soc = do_training(df_soc, FEATURES_TRAIN)
df_bio["train_subj"] = df_bio.apply(
    lambda x: get_validity(x, "/validate", params_bio), axis=1
)
df_phy["train_subj"] = df_phy.apply(
    lambda x: get_validity(x, "/validate", params_phy), axis=1
)
df_soc["train_subj"] = df_soc.apply(
    lambda x: get_validity(x, "/validate", params_soc), axis=1
)
subject_validity = (
    df_bio["train_subj"].append(df_phy["train_subj"]).append(df_soc["train_subj"])
)
df["train_subj"] = subject_validity

### Lastly, let's look at the error rate as a function of changes to the spelling corrector
edit_distances = range(3, 4)
for edit_distance in edit_distances:
    parser.create_symspell_parser(
        edit_distance, parser.prefix_length, parser.spelling_dictionary_file
    )
    output_str = "spell_" + str(edit_distance)
    df[output_str] = df.apply(
        lambda x: get_validity(x, "/validate", FEATURES_NEW), axis=1
    )

print("Done!")
