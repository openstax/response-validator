# Quick script to test out some algorithm variants using some test data
# To run me: Be in the directory above this script and do python3 -m algo_tests.test_algo_variants

from validator import app
from validator.app import DEFAULTS, PARSER_FEATURE_DICT, validate_response, update_parameter_dictionary, get_question_data
from urllib.parse import urlencode
from collections import OrderedDict
import pandas as pd


FEATURES_OLD = OrderedDict(
    {
        "stem_word_count": 0,
        "option_word_count": 0,
        "innovation_word_count": 2.2,
        "domain_word_count": 2.5,
        "bad_word_count": -3,
        "common_word_count": .7
    }
)

FEATURES_NEW = OrderedDict(
    {
        "stem_word_count": 2.2,
        "option_word_count": 2.2,
        "innovation_word_count": 0,
        "domain_word_count": 0,
        "bad_word_count": -3,
        "common_word_count": .7
    }
)

# Configure the app
app.app.config["TESTING"] = True
client = app.app.test_client()

# Load the test data and append together
df1 = pd.read_csv('algo_tests/data/valid_test_1000.csv')
df2 = pd.read_csv('algo_tests/data/invalid_test_1000.csv')
df = df1.append(df2)

# The idea here is that we will be able to pass in the route along with any parameters and get the validity label out
def do_api_call(data, route, algo_params):
    params = {"response": data.response, "uid": data.uid}
    params.update(algo_params)
    resp = client.get(route, query_string=urlencode(params))
    return resp

data = df.loc[93].iloc[1]
feature_dict = update_parameter_dictionary(FEATURES_NEW, PARSER_FEATURE_DICT)
parser_dict = DEFAULTS
output = validate_response(data.response, data.uid, feature_dict, **parser_dict)
print(output)

vocab_dict, uid_used, has_numeric = get_question_data(data.uid)