# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import Flask, jsonify, request
from validator.utils import get_fixed_data
from validator.ml.stax_string_proc import StaxStringProc
from flask_cors import cross_origin
import pkg_resources

import nltk
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from collections import OrderedDict
import collections
import re
import time


DATA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")
app = Flask(__name__)

nltk.data.path = [pkg_resources.resource_filename("validator", "ml/corpora/nltk_data")]

# Default parameters for the response parser and validation call
PARSER_DEFAULTS = {
    "remove_stopwords": True,
    "tag_numeric": "auto",
    "spelling_correction": "auto",
    "remove_nonwords": True,
    "spell_correction_max": 10,
    "lazy_math_mode": True
}

SPELLING_CORRECTION_DEFAULTS = {
    "spell_correction_max_edit_distance": 3,
    "spell_correction_min_word_length": 5
}

# If number, feature is used and has the corresponding weight.
# A value of 0 indicates that the feature won't be computed
VALIDITY_FEATURE_DICT = collections.OrderedDict(
    {
        "stem_word_count": 0,
        "option_word_count": 0,
        "innovation_word_count": 2.2,
        "domain_word_count": 2.5,
        "bad_word_count": -3,
        "common_word_count": .7,
        "intercept": 0
    }
)

# Get the global data for the app:
#    innovation words by module,
#    domain words by subject,
#    and table linking question uid to cnxmod
df_innovation, df_domain, df_questions = get_fixed_data()
uid_set = df_questions.uid.values.tolist()
qid_set = df_questions.qid.values.tolist()

# Define common and bad vocab
with open(f"{DATA_PATH}/bad.txt") as f:
    bad_vocab = set([re.sub("\n", "", w) for w in f])

# Create the parser, initially assign default values
# (these can be overwritten during calls to process_string)
parser = StaxStringProc(
    corpora_list=[f"{DATA_PATH}/all_join.txt", f"{DATA_PATH}/question_text.txt"],
    parse_args=(
        PARSER_DEFAULTS["remove_stopwords"],
        PARSER_DEFAULTS["tag_numeric"],
        PARSER_DEFAULTS["spelling_correction"],
        PARSER_DEFAULTS["remove_nonwords"],
        PARSER_DEFAULTS["spell_correction_max"],
        SPELLING_CORRECTION_DEFAULTS["spell_correction_max_edit_distance"],
        SPELLING_CORRECTION_DEFAULTS["spell_correction_min_word_length"]
    ),
    symspell_dictionary_file=f"{DATA_PATH}/response_validator_spelling_dictionary.txt"
)

common_vocab = set(parser.all_words) | set(parser.reserved_tags)


def get_question_data_by_key(key, val):
    first_q = df_questions[df_questions[key] == val].iloc[0]
    module_id = first_q.module_id
    uid = first_q.uid
    has_numeric = df_questions[df_questions[key] == val].iloc[0].contains_number
    innovation_vocab = (
        df_innovation[df_innovation["module_id"] == module_id].iloc[0].innovation_words
    )
    subject_name = (
        df_innovation[df_innovation["module_id"] == module_id].iloc[0].subject_name
    )
    domain_vocab = (
        df_domain[df_domain["CNX Book Name"] == subject_name].iloc[0].domain_words
    )

    # A better way . . . pre-process and then just to a lookup
    question_vocab = first_q['stem_words']
    mc_vocab = first_q['mc_words']
    vocab_dict = OrderedDict(
        {
        "stem_word_count": question_vocab,
        "option_word_count": mc_vocab,
        "innovation_word_count": innovation_vocab,
        "domain_word_count": domain_vocab,
        "bad_word_count": bad_vocab,
        "common_word_count": common_vocab,
        "intercept": set()
        }
    )

    return vocab_dict, uid, has_numeric


def get_question_data(uid):
    if uid is not None:
        qid = uid.split("@")[0]
        if uid in uid_set:
            return get_question_data_by_key("uid", uid)
        elif qid in qid_set:
            return get_question_data_by_key("qid", qid)
    # no uid, or not in data sets
    default_vocab_dict = OrderedDict(
        {
        "stem_word_count": set(),
        "option_word_count": set(),
        "innovation_word_count": set(),
        "domain_word_count": set(),
        "bad_word_count": bad_vocab,
        "common_word_count": common_vocab,
        "intercept": set()
        }
    )

    return default_vocab_dict, None, None


def parse_and_classify(
    response,
    feature_weight_dict,
    feature_vocab_dict,
    remove_stopwords,
    tag_numeric,
    spelling_correction,
    remove_nonwords,
    spell_correction_limit,
):

    # Parse the students response into a word list
    response_words, num_spelling_corrections = parser.process_string_spelling_limit(
        response,
        remove_stopwords=remove_stopwords,
        tag_numeric=tag_numeric,
        correct_spelling=spelling_correction,
        kill_nonwords=remove_nonwords,
        spell_correction_max=spell_correction_limit,
    )

    # Initialize all feature counts to 0
    # Then move through the feature list in order and count iff applicable
    feature_count_dict = OrderedDict(
        {
            key: 0 for key in feature_weight_dict.keys()
        }
    )
    feature_count_dict['intercept'] = 1

    for word in response_words:
        for key in feature_weight_dict.keys():
            if feature_weight_dict[key]:
                if word in feature_vocab_dict[key]:
                    feature_count_dict[key] += 1
                    break  # This will kill the inner loop when a feature is matched


    # Group the counts together and compute an inner product with the weights
    vector = feature_count_dict.values()
    WEIGHTS = feature_weight_dict.values()
    inner_product = sum([v * w for v, w in zip(vector, WEIGHTS)])
    valid = float(inner_product) > 0

    return_dict = {
        "response": response,
        "remove_stopwords": remove_stopwords,
        "tag_numeric": tag_numeric,
        "spelling_correction_used": spelling_correction,
        "num_spelling_correction": num_spelling_corrections,
        "remove_nonwords": remove_nonwords,
        "processed_response": " ".join(response_words),
    }
    return_dict.update(feature_count_dict)
    return_dict['inner_product'] = inner_product
    return_dict['valid'] = valid
    return return_dict


def validate_response(
    response,
    uid,
    feature_weight_dict,
    remove_stopwords=PARSER_DEFAULTS["remove_stopwords"],
    tag_numeric=PARSER_DEFAULTS["tag_numeric"],
    spelling_correction=PARSER_DEFAULTS["spelling_correction"],
    remove_nonwords=PARSER_DEFAULTS["remove_nonwords"],
    spell_correction_max=PARSER_DEFAULTS["spell_correction_max"],
    lazy_math_mode=PARSER_DEFAULTS["lazy_math_mode"],
):
    """Function to estimate validity given response, uid, and parser parameters"""

    # Try to get questions-specific vocab via uid (if not found, vocab will be empty)
    #domain_vocab, innovation_vocab, has_numeric, uid_used, question_vocab, mc_vocab = get_question_data(uid)
    vocab_dict, uid_used, has_numeric = get_question_data(uid)

    # Record the input of tag_numeric and then convert in the case of 'auto'
    tag_numeric_input = tag_numeric
    tag_numeric = tag_numeric or ((tag_numeric == "auto") and has_numeric)

    if spelling_correction != "auto":
        return_dictionary = parse_and_classify(
            response,
            feature_weight_dict,
            vocab_dict,
            remove_stopwords,
            tag_numeric,
            spelling_correction,
            remove_nonwords,
            spell_correction_max,
        )
    else:
        # Check for validity without spelling correction
        return_dictionary = parse_and_classify(
            response,
            feature_weight_dict,
            vocab_dict,
            remove_stopwords,
            tag_numeric,
            False,
            remove_nonwords,
            spell_correction_max,
        )

        # If that didn't pass, re-evaluate with spelling correction turned on
        if not return_dictionary["valid"]:
            return_dictionary = parse_and_classify(
                response,
                feature_weight_dict,
                vocab_dict,
                remove_stopwords,
                tag_numeric,
                True,
                remove_nonwords,
                spell_correction_max,
            )

    return_dictionary["tag_numeric_input"] = tag_numeric_input
    return_dictionary["spelling_correction"] = spelling_correction
    return_dictionary["uid_used"] = uid_used
    return_dictionary["uid_found"] = uid_used in uid_set
    return_dictionary["lazy_math_evaluation"] = lazy_math_mode

    # If lazy_math_mode, do a lazy math check and update valid accordingly
    if lazy_math_mode and response is not None:
        resp_has_math = re.search('[\+\-\*\=\/\d]', response) is not None
        return_dictionary['valid'] = return_dictionary['valid'] or (bool(has_numeric) and resp_has_math)

    return return_dictionary


def make_tristate(var, default=True):
    if type(default) == int:
        try:
            return int(var)
        except ValueError:
            pass
        try:
            return float(var)
        except:
            pass
    if var == "auto" or type(var) == bool:
        return var
    elif var in ("False", "false", "f", "0", "None", ""):
        return False
    elif var in ("True", "true", "t", "1"):
        return True
    else:
        return default


# Defines the entry point for the api call
# Read in/preps the validity arguments and then calls validate_response
# Returns JSON dictionary
# credentials are needed so the SSO cookie can be read
@app.route("/validate", methods=("GET", "POST"))
@cross_origin(supports_credentials=True)
def validation_api_entry():
    # TODO: waiting for https://github.com/openstax/accounts-rails/pull/77
    # TODO: Add the ability to parse the features provided (using defaults as backup)
    # cookie = request.COOKIES.get('ox', None)
    # if not cookie:
    #         return jsonify({ 'logged_in': False })
    # decrypted_user = decrypt.get_cookie_data(cookie)

    # Get the route arguments . . . use defaults if not supplied
    if request.method == "POST":
        args = request.form
    else:
        args = request.args

    response = args.get("response", None)
    uid = args.get("uid", None)
    parser_params = {
        key: make_tristate(args.get(key, val), val) for key, val in PARSER_DEFAULTS.items()
    }
    feature_weight_dict = OrderedDict(
        {
            key: make_tristate(args.get(key, val), val) for key, val in VALIDITY_FEATURE_DICT.items()
        }
    )

    start_time = time.time()
    return_dictionary = validate_response(response, uid, feature_weight_dict, **parser_params)

    return_dictionary["computation_time"] = time.time() - start_time

    return jsonify(return_dictionary)

@app.route("/train", methods=("GET", "POST"))
@cross_origin(supports_credentials=True)
def validation_train():

    # Read out the parser and classifier settings from the path arguments
    if request.method == "POST":
        args = request.form
    else:
        args = request.args
    train_feature_dict = {
        key: make_tristate(args.get(key, val), val) for key, val in VALIDITY_FEATURE_DICT.items()
    }
    features_to_consider = [k for k in train_feature_dict.keys() if train_feature_dict[k]]
    if ('intecept') in features_to_consider:
        features_to_consider.remove('intercept')
    parser_params = {
        key: make_tristate(args.get(key, val), val) for key, val in PARSER_DEFAULTS.items()
    }
    cv_input = args.get('cv', 5)

    # Read in the dataframe of responses from json input
    response_df = request.json.get("response_df", None)
    response_df = pd.read_json(response_df).reset_index()

    # Parse the responses in response_df to get counts on the various word categories
    # Map the valid label of the input to the output
    output_df = response_df.apply(lambda x: validate_response(x.free_response,
                                                              x.uid,
                                                              train_feature_dict,
                                                              **parser_params
                                                              ),
                                  axis=1)
    output_df = pd.DataFrame(list(output_df))
    output_df["valid_label"] = response_df["valid_label"]
    response_df.to_csv('stupid_response.csv')
    output_df.to_csv('stupid_test.csv')

    # Do an N-fold cross validation if cv > 1.
    # Then get coefficients/intercept for the entire dataset
    lr = LogisticRegression(solver='saga',
                            max_iter=1000,
                            fit_intercept=train_feature_dict["intercept"]!=0)
    X = output_df[features_to_consider].values
    y = output_df["valid_label"].values

    validation_array = -1
    if (cv_input > 1):
        validation_array = cross_val_score(lr,
                                           X,
                                           y,
                                           cv=cv_input)
    lr.fit(X, y)
    coef = lr.coef_
    intercept = lr.intercept_[0]
    validation_score = float(np.mean(validation_array))

    # Create the return dictionary with the coefficients/intercepts as well as the parsed datafrane
    # We really don't need to the return the dataframe but it's nice for debugging!
    return_dictionary = dict(
        zip(
            features_to_consider,
            coef[0].tolist()
        )
    )
    return_dictionary["intercept"] = intercept
    return_dictionary["output_df"] = output_df.to_json()
    return_dictionary["cross_val_score"] = validation_score
    return jsonify(return_dictionary)

if __name__ == "__main__":
    app.run(debug=False)  # pragma: nocover
