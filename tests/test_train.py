import os
import pytest
import shutil
import tempfile
from urllib.parse import urlencode
from collections import OrderedDict

import pandas as pd
import numpy as np
from validator import app

os.environ["VALIDATOR_SETTINGS"] = "../tests/testing.cfg"


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


@pytest.fixture(scope="module")
def myapp():
    tmpdir = tempfile.mkdtemp()
    for filename in os.listdir("tests/data"):
        shutil.copy(os.path.join("tests/data", filename), tmpdir)
    myapp = app.create_app(DATA_DIR=tmpdir)
    yield myapp
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def data(myapp):
    np.random.seed(1000)
    from validator.validate_api import bad_vocab, common_vocab, get_question_data
    df = myapp.df
    with myapp.app_context():
        question_data = df["questions"][df["questions"]["uid"] == "9@7"].iloc[0]
        stem_vocab = question_data["stem_words"]
        mc_vocab = question_data["mc_words"]
        vocab_set = get_question_data(question_data.uid)[0]
        domain_vocab = vocab_set["domain_word_count"]
        innovation_vocab = vocab_set["innovation_word_count"]
    vocab_dict = OrderedDict(
        {
            "question_data": question_data,
            "stem": stem_vocab,
            "mc": mc_vocab,
            "bad": bad_vocab,
            "common": common_vocab,
            "domain": domain_vocab,
            "innovation": innovation_vocab,
        }
    )
    yield vocab_dict


@pytest.fixture(scope="module")
def client(myapp):
    myapp.config["TESTING"] = True
    client = myapp.test_client()
    yield client


def test_train_stem_option(client, data):
    """Training with feature set 1"""
    """Make a fake dataframe with known weights. See if estimation is close(ish)"""

    N_resp = 20
    N_words = 10
    weights = OrderedDict({"stem": 1, "mc": 2, "bad": -2, "common": 0})
    #    vocab_dict = OrderedDict(
    #        {"stem": stem_vocab, "mc": mc_vocab, "bad": bad_vocab, "common": common_vocab}
    #    )

    weight_vect = np.array(list(weights.values()))

    uid = data["question_data"].uid
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
            " ".join([np.random.choice(list(data[k])) for k in word_types])
        )
    type_count = [
        np.array([r.count(t) for t in list(weights.keys())]) for r in responses_type
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
    output_df = pd.DataFrame.from_dict(out["output_df"])

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
    assert output_df["common_word_count"].sum() > 0

    # Assert that there exists a valid feature_weight_set_id
    assert type(out['feature_weight_set_id']) == str

    # Verify that values returned from the call to train match the /datasets/feature_weights path
    resp = client.get(f"/datasets/feature_weights/{out['feature_weight_set_id']}")
    for key in resp.json.keys():
        if resp.json[key] != 0:
            assert resp.json[key] == out[key]
        elif key == "intercept":
            assert resp.json[key] == 0
        else:
            assert key not in out.keys()

    # FIXME: Comparison between results: values are correct
#     import pdb;
#     pdb.set_trace()
#     for key in resp.json.keys():
#         result = resp.json[key]
#         expected = standard_results[key]
# #        TestCase.assertAlmostEqual(results, avg, delta=0.35)
#         assert abs(expected - result) <= 0.1

def test_train_domain_innovation(client, data):
    """Training with feature set 1"""
    """Make a fake dataframe with known weights. See if estimation is close(ish)"""
    N_resp = 20
    N_words = 10
    weights = OrderedDict({"domain": 1, "innovation": 2, "bad": -2, "common": 0})
    # vocab_dict = OrderedDict(
    #     {
    #         "domain": domain_vocab,
    #         "innovation": innovation_vocab,
    #         "bad": bad_vocab,
    #         "common": common_vocab,
    #     }
    # )
    weight_vect = np.array(list(weights.values()))

    uid = data["question_data"].uid
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
            " ".join([np.random.choice(list(data[k])) for k in word_types])
        )
    type_count = [
        np.array([r.count(t) for t in list(weights.keys())]) for r in responses_type
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
    output_df = pd.DataFrame.from_dict(out["output_df"])

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

    # Assert that there exists a valid feature_weight_set_id
    assert type(out['feature_weight_set_id']) == str

    # Verify that values returned from the call to train match the /datasets/feature_weights path
    resp = client.get(f"/datasets/feature_weights/{out['feature_weight_set_id']}")
    for key in resp.json.keys():
        if resp.json[key] != 0:
            assert resp.json[key] == out[key]
        elif key == "intercept":
            assert resp.json[key] == 0
        else:
            assert key not in out.keys()

