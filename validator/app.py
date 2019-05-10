# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import Flask, jsonify, request
from validator.utils import get_fixed_data
from validator.ml.stax_string_proc import StaxStringProc
from nltk.corpus import words
import re
import time
from flask_cors import cross_origin

import pkg_resources

DATA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")
app = Flask(__name__)

# Default parameters for the response parser, and validation call
DEFAULTS = {
    "remove_stopwords": True,
    "tag_numeric": "auto",
    "spelling_correction": "auto",
    "remove_nonwords": True,
    "spell_correction_max": 10,
}
# Valid response has to have a positive final weighted count
# weights are:
#  bad_words, domain_words, innovation_words, common_words
WEIGHTS = [-3, 2.5, 2.2, 0.7]

# Get the global data for the app:
#    innovation words by module,
#    domain words by subject,
#    and table linking question uid to cnxmod
df_innovation, df_domain, df_questions = get_fixed_data()

# question_set = df_questions.uid.values.tolist()
uid_set = df_questions.uid.values.tolist()
qid_set = df_questions.qid.values.tolist()

# Define common and bad vocab
with open(f"{DATA_PATH}/bad.txt") as f:
    bad_vocab = set([re.sub("\n", "", w) for w in f])

# Create the parser, initially assign default values
# (these can be overwritten during calls to process_string)
parser = StaxStringProc(
    corpora_list=[f"{DATA_PATH}/all_join.txt"],
    parse_args=(
        DEFAULTS["remove_stopwords"],
        DEFAULTS["tag_numeric"],
        DEFAULTS["spelling_correction"],
        DEFAULTS["remove_nonwords"],
        DEFAULTS["spell_correction_max"],
    ),
)

common_vocab = set(words.words()) | set(parser.reserved_tags)


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
    return domain_vocab, innovation_vocab, has_numeric, uid


def get_question_data(uid):
    if uid is not None:
        qid = uid.split("@")[0]
        if uid in uid_set:
            return get_question_data_by_key("uid", uid)
        elif qid in qid_set:
            return get_question_data_by_key("qid", qid)
    # no uid, or not in data sets
    return set(), set(), None, None


def parse_and_classify(
    response,
    innovation_vocab,
    domain_vocab,
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

    # Compute intersection cardinality with each of the sets of interest
    bad_count = domain_count = innovation_count = common_count = 0
    for word in response_words:
        if word in bad_vocab:
            bad_count += 1
        elif word in innovation_vocab:
            innovation_count += 1
        elif word in domain_vocab:
            domain_count += 1
        elif word in common_vocab:
            common_count += 1

    # Group the counts together and compute an inner product with the weights
    vector = [bad_count, domain_count, innovation_count, common_count]
    inner_product = sum([v * w for v, w in zip(vector, WEIGHTS)])
    valid = float(inner_product) > 0

    return {
        "response": response,
        "remove_stopwords": remove_stopwords,
        "tag_numeric": tag_numeric,
        "spelling_correction_used": spelling_correction,
        "num_spelling_correction": num_spelling_corrections,
        "remove_nonwords": remove_nonwords,
        "processed_response": " ".join(response_words),
        "bad_word_count": bad_count,
        "domain_word_count": domain_count,
        "innovation_word_count": innovation_count,
        "common_word_count": common_count,
        "inner_product": inner_product,
        "valid": valid,
    }


def validate_response(
    response,
    uid,
    remove_stopwords=DEFAULTS["remove_stopwords"],
    tag_numeric=DEFAULTS["tag_numeric"],
    spelling_correction=DEFAULTS["spelling_correction"],
    remove_nonwords=DEFAULTS["remove_nonwords"],
    spell_correction_max=DEFAULTS["spell_correction_max"],
):
    """Function to estimate validity given response, uid, and parser parameters"""

    # Try to get questions-specific vocab via uid (if not found, vocab will be empty)
    domain_vocab, innovation_vocab, has_numeric, uid_used = get_question_data(uid)

    # Record the input of tag_numeric and then convert in the case of 'auto'
    tag_numeric_input = tag_numeric
    tag_numeric = tag_numeric or ((tag_numeric == "auto") and has_numeric)

    if spelling_correction != "auto":
        return_dictionary = parse_and_classify(
            response,
            innovation_vocab,
            domain_vocab,
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
            innovation_vocab,
            domain_vocab,
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
                innovation_vocab,
                domain_vocab,
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

    return return_dictionary


def make_tristate(var, default=True):
    if type(default) == int:
        try:
            return int(var)
        except ValueError:
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
    params = {
        key: make_tristate(args.get(key, val), val) for key, val in DEFAULTS.items()
    }

    start_time = time.time()
    return_dictionary = validate_response(response, uid, **params)

    return_dictionary["computation_time"] = time.time() - start_time

    return jsonify(return_dictionary)


if __name__ == "__main__":
    app.run(debug=False)  # pragma: nocover
