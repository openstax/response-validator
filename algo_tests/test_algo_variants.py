# Quick script to test out some algorithm variants using some test data
# To run me: Be in the directory above this script and do python3 -m algo_tests.test_algo_variants

from validator import app
from urllib.parse import urlencode
from collections import OrderedDict
import pandas as pd
from validator.app import parser
from sklearn.metrics import confusion_matrix


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
df = df1.append(df2).reset_index()

# The idea here is that we will be able to pass in the route along with any parameters and get the validity label out
def do_api_call(data, route, algo_params):
    params = {"response": data.response, "uid": data.uid}
    params.update(algo_params)
    resp = client.get(route, query_string=urlencode(params))
    return resp

def get_validity(data, route, algo_params):
    resp = do_api_call(data, route, algo_params)
    valid = resp.json['valid']
    return valid

def do_training(df, train_params):
    params = {}
    params.update(train_params)
    df_train = df[['response', 'uid', 'valid_truth']].rename(columns={'response': 'free_response',
                                                                      'uid': 'uid',
                                                                      'valid_truth': 'valid_label'}).to_json()
    resp = client.get('/train', query_string=urlencode(params), json={"response_df": df_train})
    resp_dict = resp.json
    new_params = {k: resp_dict[k] for k in train_params.keys() & resp_dict.keys()}
    train_params.update(new_params)
    intercept = resp_dict['intercept']
    return new_params, intercept



print("Starting!")

#resp = do_api_call(df.iloc[0], '/validate', FEATURES_OLD)
#print(resp.json)

# Old doesn't use lazy math evaluation, new does
df['old'] = df.apply(lambda x: get_validity(x, "/validate", FEATURES_OLD), axis=1)
df['new'] = df.apply(lambda x: get_validity(x, "/validate_new", FEATURES_NEW), axis=1)

# Let's see what we can get with training (looks like cv performs about as well as hand-tuned)
new_params_opt, intercept = do_training(df, FEATURES_NEW)
app.PARSER_FEATURE_INTERCEPT = intercept
df['new_opt'] = df.apply(lambda x: get_validity(x, "/validate_new", new_params_opt), axis=1)

# Last thing . . . let's try changing the edit distance limit and examine the effect on the confusion matrix
parser.create_symspell_parser(3, 7, parser.spelling_dictionary_file)
df['new_edit3'] = df.apply(lambda x: get_validity(x, "/validate_new", FEATURES_NEW), axis=1)
parser.create_symspell_parser(4, 7, parser.spelling_dictionary_file)
df['new_edit4'] = df.apply(lambda x: get_validity(x, "/validate_new", FEATURES_NEW), axis=1)

confusion_matrix(df['valid_truth'], df['new'])
confusion_matrix(df['valid_truth'], df['new_edit3'])   # works pretty good
confusion_matrix(df['valid_truth'], df['new_edit4'])

print("Done!")