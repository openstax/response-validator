# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import jsonify, request, Blueprint, current_app
from flask_cors import cross_origin

from .read_api import InvalidUsage, handle_invalid_usage
from .utils import make_tristate
from .ml.stax_string_proc import StaxStringProc


import pkg_resources

from collections import OrderedDict
import re
import time

from . import __version__


CORPORA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")

# Define bad vocab
with open(f"{CORPORA_PATH}/bad.txt") as f:
    bad_vocab = set(f.read().split())

DEFAULT_FEATURE_WEIGHTS = {}
PARSER_DEFAULTS = {}
parser = None
common_vocab = set()

bp = Blueprint("validate_api", __name__, url_prefix="/")
bp.register_error_handler(InvalidUsage, handle_invalid_usage)


@bp.record_once
def setup_parse_and_data(setup_state):
    global DEFAULT_FEATURE_WEIGHTS, PARSER_DEFAULTS, parser, common_vocab

    PARSER_DEFAULTS = setup_state.app.config["PARSER_DEFAULTS"]
    SPELLING_CORRECTION_DEFAULTS = setup_state.app.config[
        "SPELLING_CORRECTION_DEFAULTS"
    ]
    # DEFAULT_FEATURE_WEIGHTS = setup_state.app.config["DEFAULT_FEATURE_WEIGHTS"]

    # Create the parser, initially assign default values
    # (these can be overwritten during calls to process_string)
    parser = StaxStringProc(
        corpora_list=[
            f"{CORPORA_PATH}/all_join.txt",
            f"{CORPORA_PATH}/question_text.txt",
        ],
        parse_args=(
            PARSER_DEFAULTS["remove_stopwords"],
            PARSER_DEFAULTS["tag_numeric"],
            PARSER_DEFAULTS["spelling_correction"],
            PARSER_DEFAULTS["remove_nonwords"],
            PARSER_DEFAULTS["spell_correction_max"],
            SPELLING_CORRECTION_DEFAULTS["spell_correction_max_edit_distance"],
            SPELLING_CORRECTION_DEFAULTS["spell_correction_min_word_length"],
        ),
        symspell_dictionary_file=(
            f"{CORPORA_PATH}/response_validator_spelling_dictionary.txt"
        ),
    )

    common_vocab = set(parser.all_words) | set(parser.reserved_tags)


def get_question_data_by_key(key, val):
    datasets = current_app.datasets
    # FIXME - should use all the questions and combine associated pages
    # FIXME - last_q works better because of some dirty data getting through
    # that has innovation pages but not the exact book those pages are from
    last_q = datasets["questions"][datasets["questions"][key] == val].iloc[-1]
    module_id = last_q.cvuid
    uid = last_q.uid
    has_numeric = bool(last_q.contains_number)
    innovation_vocab = (
        datasets["innovation"][datasets["innovation"]["cvuid"] == module_id]
        .iloc[0]
        .innovation_words
    )
    vuid = module_id.split(":")[0]

    domain_vocab_df = datasets["domain"][datasets["domain"]["vuid"] == vuid]
    if domain_vocab_df.empty:
        domain_vocab = set()
    else:
        domain_vocab = domain_vocab_df.iloc[-1].domain_words

    # A better way . . . pre-process and then just to a lookup
    question_vocab = last_q["stem_words"]
    mc_vocab = last_q["mc_words"]
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

    return vocab_dict, uid, has_numeric, vuid


def get_question_data(uid):
    if uid is not None:
        qid = uid.split("@")[0]
        if uid in current_app.qids["uid"]:
            return get_question_data_by_key("uid", uid)
        elif qid in current_app.qids["qid"]:
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

    return default_vocab_dict, uid, None, None


def parse_and_classify(
    response,
    feature_vocab_dict,
    remove_stopwords,
    tag_numeric,
    spelling_correction,
    remove_nonwords,
    spell_correction_limit,
    feature_weights_id,
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

    # Fetch feature weights by ID
    feature_weight_dict = current_app.datasets["feature_weights"][feature_weights_id]

    # Initialize all feature counts to 0
    # ORDER OF KEYS in feature_weight_dict is preserved, and matters!
    # Then move through the feature list in order and count iff applicable
    feature_count_dict = OrderedDict.fromkeys(feature_weight_dict.keys(), 0)
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
        "tag_numeric": bool(tag_numeric),
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
    remove_stopwords=None,
    tag_numeric=None,
    spelling_correction=None,
    remove_nonwords=None,
    spell_correction_max=None,
    lazy_math_mode=None,
    feature_weights_id=None,
):
    """Function to estimate validity given response, uid, and parser parameters"""

    if remove_stopwords is None:
        remove_stopwords = PARSER_DEFAULTS["remove_stopwords"]
    if tag_numeric is None:
        tag_numeric = PARSER_DEFAULTS["tag_numeric"]
    if spelling_correction is None:
        spelling_correction = PARSER_DEFAULTS["spelling_correction"]
    if remove_nonwords is None:
        remove_nonwords = PARSER_DEFAULTS["remove_nonwords"]
    if spell_correction_max is None:
        spell_correction_max = PARSER_DEFAULTS["spell_correction_max"]
    if lazy_math_mode is None:
        lazy_math_mode = PARSER_DEFAULTS["lazy_math_mode"]

    # Try to get questions-specific vocab via uid (if not found, vocab will be empty)
    # domain_vocab, innovation_vocab, has_numeric, uid_used, question_vocab,
    #  mc_vocab = get_question_data(uid)
    vocab_dict, uid_used, has_numeric, vuid = get_question_data(uid)
    if vuid is None:
        feature_weights_id = current_app.datasets["feature_weights"]["default_id"]

    # When there's no feature_weights id passed from user API
    # Test if feature_weights id exists for the book
    # If so, fetch from datasets["domain"]
    # Otherwise use system default
    if feature_weights_id is None:
        datasets = current_app.datasets
        domain_vocab_df = datasets["domain"][datasets["domain"]["vuid"] == vuid]
        feature_weights_id = domain_vocab_df.iloc[0]["feature_weights_id"]
        if feature_weights_id == "":
            feature_weights_id = current_app.datasets["feature_weights"]["default_id"]

    # Record the input of tag_numeric and then convert in the case of "auto"
    # The conversion is thus: if auto, we will tag numeric if has_numeric is not False
    # This means that has_numeric = True and has_numeric is None (question not found) will be True
    tag_numeric_input = tag_numeric
    if tag_numeric == "auto":
        tag_numeric = has_numeric is not False

    if spelling_correction != "auto":
        return_dictionary = parse_and_classify(
            response,
            vocab_dict,
            remove_stopwords,
            tag_numeric,
            spelling_correction,
            remove_nonwords,
            spell_correction_max,
            feature_weights_id,
        )
    else:
        # Check for validity without spelling correction
        return_dictionary = parse_and_classify(
            response,
            vocab_dict,
            remove_stopwords,
            tag_numeric,
            False,
            remove_nonwords,
            spell_correction_max,
            feature_weights_id,
        )

        # If that didn't pass, re-evaluate with spelling correction turned on
        if not return_dictionary["valid"]:
            return_dictionary = parse_and_classify(
                response,
                vocab_dict,
                remove_stopwords,
                tag_numeric,
                True,
                remove_nonwords,
                spell_correction_max,
                feature_weights_id,
            )

    return_dictionary["tag_numeric_input"] = tag_numeric_input
    return_dictionary["spelling_correction"] = spelling_correction
    return_dictionary["uid_used"] = uid_used
    return_dictionary["uid_found"] = uid_used in current_app.qids["uid"]
    return_dictionary["lazy_math_evaluation"] = lazy_math_mode

    # If lazy_math_mode, do a lazy math check and update valid accordingly
    if lazy_math_mode and response is not None:
        resp_has_math = re.search(r"[\+\-\*\=\/\d]", response) is not None
        return_dictionary["valid"] = return_dictionary["valid"] or (
            bool(has_numeric) and resp_has_math
        )

    return return_dictionary


# Defines the entry point for the api call
# Read in/preps the validity arguments and then calls validate_response
# Returns JSON dictionary
# credentials are needed so the SSO cookie can be read
@bp.route("/validate", methods=("GET", "POST"))
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

    feature_weights_set_id = args.get(
        "feature_weights_set_id", current_app.datasets["feature_weights"]["default_id"]
    )

    if feature_weights_set_id not in current_app.datasets["feature_weights"]:
        raise InvalidUsage("feature_weights_set_id not found", status_code=404)

    parser_params = {
        key: make_tristate(args.get(key, val), val)
        for key, val in PARSER_DEFAULTS.items()
    }

    start_time = time.time()
    return_dictionary = validate_response(
        response, uid, feature_weights_id=feature_weights_set_id, **parser_params
    )

    return_dictionary["feature_weights"] = current_app.datasets["feature_weights"][
        feature_weights_set_id
    ]

    return_dictionary["computation_time"] = time.time() - start_time

    return_dictionary["version"] = __version__

    return jsonify(return_dictionary)
