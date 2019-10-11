# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import Flask, jsonify, request
from flask_cors import cross_origin
from uuid import UUID

from validator.utils import get_fixed_data, write_fixed_data
from validator.ecosystem_importer import EcosystemImporter
from validator.ml.stax_string_proc import StaxStringProc

import json
import pkg_resources

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from collections import OrderedDict
import collections
import re
import time

from . import __version__, _version

start_time = time.ctime()

DATA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")
app = Flask(__name__)

# Default parameters for the response parser and validation call
PARSER_DEFAULTS = {
    "remove_stopwords": True,
    "tag_numeric": "auto",
    "spelling_correction": "auto",
    "remove_nonwords": True,
    "spell_correction_max": 10,
    "lazy_math_mode": True,
}

SPELLING_CORRECTION_DEFAULTS = {
    "spell_correction_max_edit_distance": 3,
    "spell_correction_min_word_length": 5,
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
        "common_word_count": 0.7,
        "intercept": 0,
    }
)

# Get the global data for the app:
#    innovation words by module,
#    domain words by subject,
#    and table linking question uid to cnxmod
df_innovation, df_domain, df_questions = get_fixed_data()
qids = {}
for idcol in ("uid", "qid"):
    if idcol in df_questions:
        qids[idcol] = df_questions[idcol].values.tolist()
    else:
        qids[idcol] = []

# Instantiate the ecosystem importer that will be used by the import route
ecosystem_importer = EcosystemImporter(
    common_vocabulary_filename=f"{DATA_PATH}/big.txt"
)

# Define bad vocab
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
        SPELLING_CORRECTION_DEFAULTS["spell_correction_min_word_length"],
    ),
    symspell_dictionary_file=f"{DATA_PATH}/response_validator_spelling_dictionary.txt",
)

common_vocab = set(parser.all_words) | set(parser.reserved_tags)


def update_fixed_data(df_domain_, df_innovation_, df_questions_):

    # AEW: I feel like I am sinning against nature here . . .
    # Do we need to store these in a Redis cache or db???
    # This was all well and good before we ever tried to modify things
    global df_domain, df_innovation, df_questions

    # Remove any entries from the domain, innovation, and question dataframes
    # that are duplicated by the new data
    book_id = df_domain_.iloc[0]["vuid"]
    if "vuid" in df_domain.columns:
        df_domain = df_domain[df_domain["vuid"] != book_id]
    if "cvuid" in df_domain.columns:
        df_innovation = df_innovation[
            ~(df_innovation["cvuid"].star.startswith(book_id))
        ]
    uids = df_questions_["uid"].unique()
    if "uid" in df_questions.columns:
        df_questions = df_questions[
            ~(
                df_questions["uid"].isin(uids)
                & df_questions["cvuid"].str.startswith(book_id)
            )
        ]

    # Now append the new dataframes to the in-memory ones
    df_domain = df_domain.append(df_domain_)
    df_innovation = df_innovation.append(df_innovation_)
    df_questions = df_questions.append(df_questions_)

    # Finally, write the updated dataframes to disk and declare victory
    write_fixed_data(df_domain, df_innovation, df_questions)


def get_question_data_by_key(key, val):
    first_q = df_questions[df_questions[key] == val].iloc[0]
    module_id = first_q.cvuid
    uid = first_q.uid
    has_numeric = df_questions[df_questions[key] == val].iloc[0].contains_number
    innovation_vocab = (
        df_innovation[df_innovation["cvuid"] == module_id].iloc[0].innovation_words
    )
    vuid = module_id.split(":")[0]
    domain_vocab = df_domain[df_domain["vuid"] == vuid].iloc[0].domain_words

    # A better way . . . pre-process and then just to a lookup
    question_vocab = first_q["stem_words"]
    mc_vocab = first_q["mc_words"]
    vocab_dict = OrderedDict(
        {
            "stem_word_count": question_vocab,
            "option_word_count": mc_vocab,
            "innovation_word_count": innovation_vocab,
            "domain_word_count": domain_vocab,
            "bad_word_count": bad_vocab,
            "common_word_count": common_vocab,
            "intercept": set(),
        }
    )

    return vocab_dict, uid, has_numeric


def get_question_data(uid):
    if uid is not None:
        qid = uid.split("@")[0]
        if uid in qids["uid"]:
            return get_question_data_by_key("uid", uid)
        elif qid in qids["qid"]:
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
            "intercept": set(),
        }
    )

    return default_vocab_dict, uid, None


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
    feature_count_dict = OrderedDict({key: 0 for key in feature_weight_dict.keys()})
    feature_count_dict["intercept"] = 1

    for word in response_words:
        for key in feature_weight_dict.keys():
            if feature_weight_dict[key]:
                if word in feature_vocab_dict[key]:
                    feature_count_dict[key] += 1
                    break  # only count each word as first feature matched (key order matters!)

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
    return_dict["inner_product"] = inner_product
    return_dict["valid"] = valid
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
    # domain_vocab, innovation_vocab, has_numeric, uid_used, question_vocab,
    #  mc_vocab = get_question_data(uid)
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
    return_dictionary["uid_found"] = uid_used in qids["uid"]
    return_dictionary["lazy_math_evaluation"] = lazy_math_mode

    # If lazy_math_mode, do a lazy math check and update valid accordingly
    if lazy_math_mode and response is not None:
        resp_has_math = re.search(r"[\+\-\*\=\/\d]", response) is not None
        return_dictionary["valid"] = return_dictionary["valid"] or (
            bool(has_numeric) and resp_has_math
        )

    return return_dictionary


def make_tristate(var, default=True):
    if type(default) == int:
        try:
            return int(var)
        except ValueError:
            pass
        try:
            return float(var)
        except:  # noqa
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
        key: make_tristate(args.get(key, val), val)
        for key, val in PARSER_DEFAULTS.items()
    }
    feature_weight_dict = OrderedDict(
        {
            key: make_tristate(args.get(key, val), val)
            for key, val in VALIDITY_FEATURE_DICT.items()
        }
    )

    start_time = time.time()
    return_dictionary = validate_response(
        response, uid, feature_weight_dict, **parser_params
    )

    return_dictionary["computation_time"] = time.time() - start_time

    return_dictionary["version"] = __version__

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
        key: make_tristate(args.get(key, val), val)
        for key, val in VALIDITY_FEATURE_DICT.items()
    }
    features_to_consider = [
        k for k in train_feature_dict.keys() if train_feature_dict[k]
    ]
    if ("intecept") in features_to_consider:
        features_to_consider.remove("intercept")
    parser_params = {
        key: make_tristate(args.get(key, val), val)
        for key, val in PARSER_DEFAULTS.items()
    }
    cv_input = args.get("cv", 5)

    # Read in the dataframe of responses from json input
    response_df = request.json.get("response_df", None)
    response_df = pd.read_json(response_df).reset_index()

    # Parse the responses in response_df to get counts on the various word categories
    # Map the valid label of the input to the output
    output_df = response_df.apply(
        lambda x: validate_response(
            x.free_response, x.uid, train_feature_dict, **parser_params
        ),
        axis=1,
    )
    output_df = pd.DataFrame(list(output_df))
    output_df["valid_label"] = response_df["valid_label"]

    # Do an N-fold cross validation if cv > 1.
    # Then get coefficients/intercept for the entire dataset
    lr = LogisticRegression(
        solver="saga", max_iter=1000, fit_intercept=train_feature_dict["intercept"] != 0
    )
    X = output_df[features_to_consider].values
    y = output_df["valid_label"].values

    validation_array = -1
    if cv_input > 1:
        validation_array = cross_val_score(lr, X, y, cv=cv_input)
    lr.fit(X, y)
    coef = lr.coef_
    intercept = lr.intercept_[0]
    validation_score = float(np.mean(validation_array))

    # Create the return dictionary with the coefficients/intercepts as well as
    # the parsed datafrane We really don't need to the return the dataframe but
    # it's nice for debugging!

    return_dictionary = dict(zip(features_to_consider, coef[0].tolist()))
    return_dictionary["intercept"] = intercept
    return_dictionary["output_df"] = output_df.to_json()
    return_dictionary["cross_val_score"] = validation_score
    return jsonify(return_dictionary)


@app.route("/import", methods=["POST"])
@cross_origin(supports_credentials=True)
def import_ecosystem():

    # Extract arguments for the ecosystem to import
    # Either be a file location, YAML-as-string, or book_id and list of question uids

    yaml_string = request.files["file"].read()
    if "file" in request.files:
        df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_yaml_string(
            yaml_string
        )

    elif request.json is not None:
        yaml_filename = request.json.get("filename", None)
        yaml_string = request.json.get("yaml_string", None)
        book_id = request.json.get("book_id", None)
        exercise_list = request.json.get("question_list", None)

        if yaml_filename:
            df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_yaml_file(
                yaml_filename
            )
        elif yaml_string:
            df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_yaml_string(
                yaml_string
            )
        elif book_id and exercise_list:
            df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_content(
                book_id, exercise_list
            )

        else:
            return jsonify(
                {
                    "msg": "Could not process input. Provide either"
                    " a location of a YAML file,"
                    " a string of YAML content,"
                    " or a book_id and question_list"
                }
            )

    update_fixed_data(df_domain_, df_innovation_, df_questions_)

    return jsonify({"msg": "Ecosystem successfully imported"})


# API read routes


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route("/datasets")
def datasets_index():
    return jsonify(["books", "questions"])  # FIXME , "feature_coefficients"])


def _books_json(include_vocabs=True):
    data = df_domain[["book_name", "vuid"]].rename({"book_name": "name"}, axis=1)
    if include_vocabs:
        data["vocabularies"] = [["domain", "innovation", "questions"]] * len(data)
    return data.to_dict(orient="records")


def _validate_version(ver):
    parts = ver.split(".")
    try:
        [int(part) for part in parts]
    except ValueError:
        raise InvalidUsage("Bad version")


def _validate_vuid(vuid, vuid_type="book"):
    try:
        v_uuid, ver = vuid.split("@")
    except ValueError:
        raise InvalidUsage("Need uuid and version")

    try:
        _ = UUID(v_uuid)
    except ValueError:
        raise InvalidUsage(f"Not a valid uuid for {vuid_type}")

    _validate_version(ver)


@app.route("/datasets/books")
def books_index():
    return jsonify(_books_json())


@app.route("/datasets/books/<vuid>")
def fetch_book(vuid):
    data = df_domain[df_domain["vuid"] == vuid][["book_name", "vuid"]].rename(
        {"book_name": "name"}, axis=1
    )
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    data["vocabularies"] = [["domain", "innovation", "questions"]] * len(data)
    page_list = (
        df_innovation[df_innovation["cvuid"].str.startswith(vuid)]
        .cvuid.str.split(":", expand=True)[1]
        .tolist()
    )
    data_json = data.to_dict(orient="records")[0]
    data_json["pages"] = page_list
    return jsonify(data_json)


@app.route("/datasets/books/<vuid>/pages")
def fetch_page_list(vuid):
    book = df_innovation[df_innovation["cvuid"].str.startswith(vuid)]
    if book.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    page_list = book.cvuid.str.split(":", expand=True)[1].tolist()
    return jsonify(page_list)


@app.route("/datasets/books/<vuid>/pages/<pvuid>")
def fetch_page(vuid, pvuid):
    innovation = df_innovation[df_innovation["cvuid"] == ":".join((vuid, pvuid))][
        "innovation_words"
    ]
    if innovation.empty:
        _validate_vuid(vuid)
        _validate_vuid(pvuid, vuid_type="page")
        raise InvalidUsage("No such book or page", status_code=404)

    questions = (
        df_questions[df_questions["cvuid"] == ":".join((vuid, pvuid))][
            ["uid", "mc_words", "stem_words"]
        ]
        .rename({"uid": "exercise_uid", "mc_words": "option_words"}, axis=1)
        .to_json(orient="records")
    )

    data = {
        "cvuid": ":".join((vuid, pvuid)),
        "vocabularies": {
            "innovation": list(innovation.iloc[0]),
            "questions": json.loads(questions),
        },
    }
    return jsonify(data)


@app.route("/datasets/books/<vuid>/vocabularies")
def fetch_vocabs(vuid):
    return jsonify(["domain", "innovation", "questions"])


@app.route("/datasets/books/<vuid>/vocabularies/domain")
def fetch_domain(vuid):
    data = df_domain[df_domain["vuid"] == vuid]["domain_words"]
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    return jsonify(list(data.tolist()[0]))


@app.route("/datasets/books/<vuid>/vocabularies/innovation")
def fetch_innovation(vuid):
    data = df_innovation[df_innovation["cvuid"].str.startswith(vuid)][
        ["cvuid", "innovation_words"]
    ]
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    data["page_vuid"] = data.cvuid.str.split(":", expand=True)[1]
    data['innovation_words'] = data['innovation_words'].map(list)
    return jsonify(
        data[["page_vuid", "innovation_words"]].to_dict(orient="records")
    )


@app.route("/datasets/books/<vuid>/vocabularies/innovation/<pvuid>")
def fetch_page_innovation(vuid, pvuid):
    data = df_innovation[df_innovation["cvuid"] == ":".join((vuid, pvuid))][
        "innovation_words"
    ]
    if data.empty:
        _validate_vuid(vuid)
        _validate_vuid(pvuid, vuid_type="page")
        raise InvalidUsage("No such book or page", status_code=404)

    return jsonify(list(data.iloc[0]))


@app.route("/datasets/books/<vuid>/vocabularies/questions")
def fetch_questions(vuid):
    data = df_questions[df_questions["cvuid"].str.startswith(vuid)].rename(
        {"uid": "exercise_uid", "mc_words": "option_words"}, axis=1
    )
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    data["page_vuid"] = data.cvuid.str.split(":", expand=True)[1]
    pages = data.groupby("page_vuid")
    data_json = [
        {
            "page_vuid": page[0],
            "questions": json.loads(
                page[1][["exercise_uid", "option_words", "stem_words"]].to_json(
                    orient="records"
                )
            ),
        }
        for page in pages
    ]
    return jsonify(data_json)


@app.route("/datasets/books/<vuid>/vocabularies/questions/<pvuid>")
def fetch_page_questions(vuid, pvuid):
    data = df_questions[df_questions["cvuid"] == ":".join((vuid, pvuid))].rename(
        {"uid": "exercise_uid", "mc_words": "option_words"}, axis=1
    )
    if data.empty:
        _validate_vuid(vuid)
        _validate_vuid(pvuid, vuid_type="page")
        book = df_domain[df_domain["vuid"] == vuid]
        if book.empty:
            raise InvalidUsage("No such book", status_code=404)
        page = df_innovation[df_innovation["cvuid"] == ":".join((vuid, pvuid))]
        if page.empty:
            raise InvalidUsage("No such page in book", status_code=404)

        return jsonify([])

    json_data = json.loads(
        data[["exercise_uid", "option_words", "stem_words"]].to_json(orient="records")
    )
    return jsonify(json_data)


@app.route("/datasets/questions")
def questions_index():
    return jsonify(df_questions.uid.tolist())


@app.route("/datasets/questions/<uid>")
def fetch_question(uid):
    data = df_questions[df_questions["uid"] == uid].rename(
        {"uid": "exercise_uid", "mc_words": "option_words"}, axis=1
    )

    json_data = json.loads(
        data[["exercise_uid", "option_words", "stem_words"]].to_json(orient="records")
    )
    return jsonify(json_data)


@app.route("/ping")
def ping():
    return "pong"


@app.route("/status")
def status():
    global start_time
    data = {"version": _version.get_versions(), "started": start_time}

    if "vuid" in df_domain.columns:
        data["datasets"] = {"books": _books_json(include_vocabs=False)}

    return jsonify(data)


@app.route("/version")
@app.route("/rev.txt")
def simple_version():
    return __version__


if __name__ == "__main__":
    app.run(debug=False)  # pragma: nocover
