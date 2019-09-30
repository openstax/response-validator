import pytest
from urllib.parse import urlencode
from collections import OrderedDict

import pandas as pd
import numpy as np
from validator import app
from validator.app import PARSER_DEFAULTS
from validator.app import df_questions, bad_vocab, common_vocab, get_question_data

# A set of weights to use when testing things other than stem/option counts
FEATURE_SET_1 = {
    "stem_word_count": 1,
    "option_word_count": 1,
    "innovation_word_count": 0,
    "domain_word_count": 0,
    "bad_word_count": 1,
    "common_word_count": 1,
}

FEATURE_SET_2 = {
    "stem_word_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 1,
    "domain_word_count": 1,
    "bad_word_count": 1,
    "common_word_count": 1,
}

question_data = df_questions[df_questions["uid"] == "9@7"].iloc[0]
stem_vocab = question_data["stem_words"]
mc_vocab = question_data["mc_words"]
vocab_set = get_question_data(question_data.uid)[0]
domain_vocab = vocab_set["domain_word_count"]
innovation_vocab = vocab_set["innovation_word_count"]

vocab_dict = OrderedDict(
    {"stem": stem_vocab, "mc": mc_vocab, "bad": bad_vocab, "common": common_vocab}
)


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    client = app.app.test_client()
    yield client


def test_train_stem_option(client):
    """Training with feature set 1"""
    """Make a fake dataframe with known weights. See if estimation is close(ish)"""

    N_resp = 20
    N_words = 10
    weights = OrderedDict({"stem": 1, "mc": 2, "bad": -2, "common": 0})
    vocab_dict = OrderedDict(
        {"stem": stem_vocab, "mc": mc_vocab, "bad": bad_vocab, "common": common_vocab}
    )
    weight_vect = np.array(list(weights.values()))

    uid = question_data.uid
    response_validity = np.random.choice([True, False], N_resp)
    responses_type = []
    responses = []
    for r in response_validity:
        if r:
            word_types = np.random.choice(["stem", "mc", "common"], N_words).tolist()
        else:
            word_types = np.random.choice(["bad"], N_words).tolist()
        responses_type.append(word_types)
        responses.append(
            " ".join([np.random.choice(list(vocab_dict[k])) for k in word_types])
        )
    type_count = [
        np.array([r.count(t) for t in list(vocab_dict.keys())]) for r in responses_type
    ]
    ip = [np.sum(weight_vect * t) for t in type_count]
    valid = [val > 0 for val in ip]

    df = pd.DataFrame(
        {"uid": N_resp * [uid], "free_response": responses, "valid_label": valid}
    )
    df_json = df.to_json()

    # Call the train route, get response
    # Grab both the parsed out dataframe and the coefficients (+ intercept)
    r = client.get(
        "/train", query_string=urlencode(FEATURE_SET_1), json={"response_df": df_json}
    )
    out = r.json
    output_df = pd.read_json(out["output_df"])

    # Assert that the return dataframe has N_resp rows
    assert len(output_df) == N_resp

    # Assert that the bad_word_count field gets a negative value
    assert out["bad_word_count"] < 0

    # Assert that domain/innovation counts are all 0
    assert output_df["domain_word_count"].sum() == 0
    assert output_df["innovation_word_count"].sum() == 0

    # Assert that bad/common/stem/option words all have non-zero counts
    assert output_df["option_word_count"].sum() > 0
    assert output_df["stem_word_count"].sum() > 0
    assert output_df["bad_word_count"].sum() > 0
    assert output_df["common_word_count"].sum() > 0


def test_train_domain_innovation(client):
    """Training with feature set 1"""
    """Make a fake dataframe with known weights. See if estimation is close(ish)"""
    np.random.seed(1000)
    N_resp = 20
    N_words = 10
    weights = OrderedDict({"domain": 1, "innovation": 2, "bad": -2, "common": 0})
    vocab_dict = OrderedDict(
        {
            "domain": domain_vocab,
            "innovation": innovation_vocab,
            "bad": bad_vocab,
            "common": common_vocab,
        }
    )
    weight_vect = np.array(list(weights.values()))

    uid = question_data.uid
    response_validity = np.random.choice([True, False], N_resp)
    responses_type = []
    responses = []
    for r in response_validity:
        if r:
            word_types = np.random.choice(
                ["domain", "innovation", "common"], N_words
            ).tolist()
        else:
            word_types = np.random.choice(["bad"], N_words).tolist()
        responses_type.append(word_types)
        responses.append(
            " ".join([np.random.choice(list(vocab_dict[k])) for k in word_types])
        )
    type_count = [
        np.array([r.count(t) for t in list(vocab_dict.keys())]) for r in responses_type
    ]
    ip = [np.sum(weight_vect * t) for t in type_count]
    valid = [val > 0 for val in ip]

    df = pd.DataFrame(
        {"uid": N_resp * [uid], "free_response": responses, "valid_label": valid}
    )
    df_json = df.to_json()

    # Call the train route, get response
    # Grab both the parsed out dataframe and the coefficients (+ intercept)
    r = client.get(
        "/train", query_string=urlencode(FEATURE_SET_2), json={"response_df": df_json}
    )
    out = r.json
    output_df = pd.read_json(out["output_df"])

    # Assert that the return dataframe has N_resp rows
    assert len(output_df) == N_resp

    # Assert that the bad_word_count field gets a negative value
    assert out["bad_word_count"] < 0

    # Assert that stem/option counts are all 0
    assert output_df["stem_word_count"].sum() == 0
    assert output_df["option_word_count"].sum() == 0

    # Assert that bad/common/domain/innovation words all have non-zero counts
    assert output_df["domain_word_count"].sum() > 0
    assert output_df["innovation_word_count"].sum() > 0
    assert output_df["bad_word_count"].sum() > 0
    assert output_df["common_word_count"].sum() > 0
