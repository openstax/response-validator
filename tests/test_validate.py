import os
import pytest
from urllib.parse import urlencode

from validator import app


os.environ["VALIDATOR_SETTINGS"] = "../tests/testing.cfg"

myapp = app.create_app()
PARSER_DEFAULTS = myapp.config["PARSER_DEFAULTS"]

# A set of weights to use when testing things other than stem/option counts
NO_QUESTION_WEIGHT_DICT = {
    "stem_word_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7,
}

# A set of weights to use for testing stem/option counts
QUESTION_WEIGHT_DICT = {
    "stem_word_count": 1,
    "option_word_count": 1,
    "innovation_word_count": 0,
    "domain_word_count": 0,
    "bad_word_count": -3,
    "common_word_count": 0.7,
}


@pytest.fixture(scope="module")
def client():
    myapp.config["TESTING"] = True
    client = myapp.test_client()
    yield client


def test_validate_response():
    from validator.validate_api import validate_response

    expected = {
        "inner_product": 0,
        "intercept": 1,
        "lazy_math_evaluation": True,
        "num_spelling_correction": 0,
        "processed_response": "foo bar",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "foo bar",
        "spelling_correction": "auto",
        "spelling_correction_used": True,
        "tag_numeric": False,
        "tag_numeric_input": "auto",
        "uid_found": True,
        "uid_used": "100@7",
        "valid": False,
    }

    with myapp.app_context():
        res = validate_response("foo bar", "100@1", {})

    assert res == expected


def test_404(client):
    """Top level does nothing."""

    resp = client.get("/")
    assert resp.status_code == 404


def test_empty_get(client):
    """Empty string."""

    resp = client.get("/validate")
    expected = {
        "bad_word_count": 1,
        "common_word_count": 0,
        "computation_time": 0.00023221969604492188,
        "domain_word_count": 0,
        "inner_product": -3.0,
        "innovation_word_count": 0,
        "processed_response": "no_text",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": None,
        "spelling_correction": PARSER_DEFAULTS["spelling_correction"],
        "spelling_correction_used": True,
        "num_spelling_correction": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": False,
    }
    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    result = {k: result[k] for k in expected.keys()}
    assert result == expected


def test_empty_post(client):
    """Empty string (POST)."""

    resp = client.post("/validate")
    expected = {
        "bad_word_count": 1,
        "common_word_count": 0,
        "computation_time": 0.00023221969604492188,
        "domain_word_count": 0,
        "inner_product": -3.0,
        "innovation_word_count": 0,
        "processed_response": "no_text",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": None,
        "spelling_correction": PARSER_DEFAULTS["spelling_correction"],
        "spelling_correction_used": True,
        "num_spelling_correction": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": False,
    }
    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    result = {k: result[k] for k in expected.keys()}
    assert result == expected


def test_non_words(client):
    """Just well-known not-a-word"""

    params = {"response": "idk asdf lol n/a"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 3,
        "common_word_count": 0,
        "computation_time": 0.015013456344604492,
        "domain_word_count": 0,
        "inner_product": -8.3,
        "innovation_word_count": 0,
        "num_spelling_correction": 0,
        "processed_response": "nonsense_word nonsense_word nonsense_word math_type",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "idk asdf lol n/a",
        "spelling_correction": "auto",
        "spelling_correction_used": True,
        "tag_numeric": "auto",
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": False,
    }

    result = resp.json
    assert result["computation_time"] != 0
    assert result["common_word_count"] == expected["common_word_count"]
    assert result["bad_word_count"] == expected["bad_word_count"]
    assert result["valid"] == expected["valid"]
    # expected["computation_time"] = result["computation_time"]
    # assert expected.items() <= result.items()


def test_simple_words(client):
    """Just well-known not-a-word"""

    params = {"response": "Here is my response"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 1,
        "computation_time": 0.0004382133483886719,
        "domain_word_count": 0,
        "inner_product": 0.7,
        "innovation_word_count": 0,
        "processed_response": "response",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "Here is my response",
        "spelling_correction": PARSER_DEFAULTS["spelling_correction"],
        "spelling_correction_used": False,
        "num_spelling_correction": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": True,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert expected.items() <= result.items()


def test_domain_words(client):
    """Word in the domain of the exercise (the book)"""

    params = {"response": "echinacea chemiosmosis", "uid": "1340@1"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 0,
        "computation_time": 0.008275270462036133,
        "domain_word_count": 1,
        "inner_product": 4.7,
        "innovation_word_count": 1,
        "intercept": 1,
        "lazy_math_evaluation": True,
        "num_spelling_correction": 0,
        "option_word_count": 0,
        "processed_response": "echinacea chemiosmosis",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "echinacea chemiosmosis",
        "spelling_correction": "auto",
        "spelling_correction_used": False,
        "stem_word_count": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": True,
        "uid_used": "1340@6",
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_innovation_words(client):
    """A word in the innovation list of the exercise"""

    params = {"response": "1.0 echinacea cytosol", "uid": "290@1"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 1,
        "computation_time": 0.0060193538665771484,
        "domain_word_count": 1,
        "inner_product": 5.4,
        "innovation_word_count": 1,
        "intercept": 1,
        "lazy_math_evaluation": True,
        "num_spelling_correction": 0,
        "option_word_count": 0,
        "processed_response": "numeric_type_float echinacea cytosol",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "1.0 echinacea cytosol",
        "spelling_correction": "auto",
        "spelling_correction_used": False,
        "stem_word_count": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": True,
        "uid_used": "290@7",
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_numeric_words(client):
    """Various numerics"""

    params = {"response": "0 23 -3 1.2 IV", "tag_numeric": True}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 5,
        "computation_time": 0.000492095947265625,
        "domain_word_count": 0,
        "inner_product": 3.5,
        "innovation_word_count": 0,
        "intercept": 1,
        "lazy_math_evaluation": True,
        "num_spelling_correction": 0,
        "option_word_count": 0,
        "processed_response": "numeric_type_0 numeric_type_int "
        "numeric_type_int numeric_type_float numeric_type_roman",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "0 23 -3 1.2 IV",
        "spelling_correction": "auto",
        "spelling_correction_used": False,
        "stem_word_count": 0,
        "tag_numeric": True,
        "tag_numeric_input": True,
        "uid_found": False,
        "uid_used": None,
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected["tag_numeric"] == result["tag_numeric"]
    assert expected.items() <= result.items()


def test_no_spelling_correction(client):
    """Various numerics"""

    params = {"response": "This is my respones", "spelling_correction": False}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 1,
        "common_word_count": 0,
        "computation_time": 0.0007781982421875,
        "domain_word_count": 0,
        "inner_product": -3.0,
        "innovation_word_count": 0,
        "processed_response": "nonsense_word",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "This is my respones",
        "spelling_correction": False,
        "spelling_correction_used": False,
        "num_spelling_correction": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": False,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_auto_spelling_correction_invalid(client):
    """Various numerics"""

    params = {"response": "This is my respones", "spelling_correction": "auto"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 1,
        "computation_time": 0.0010485649108886719,
        "domain_word_count": 0,
        "inner_product": 0.7,
        "innovation_word_count": 0,
        "processed_response": "response",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "This is my respones",
        "spelling_correction": "auto",
        "spelling_correction_used": True,
        "num_spelling_correction": 1,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_auto_spelling_correction_valid(client):
    """Various numerics"""

    params = {"response": "This is my response", "spelling_correction": "auto"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 1,
        "computation_time": 0.0010485649108886719,
        "domain_word_count": 0,
        "inner_product": 0.7,
        "innovation_word_count": 0,
        "processed_response": "response",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "This is my response",
        "spelling_correction": "auto",
        "spelling_correction_used": False,
        "num_spelling_correction": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_auto_spelling_correction_limit_3(client):
    """Various numerics"""

    params = {
        "response": " ".join(["respones"] * 10),
        "spelling_correction": "auto",
        "spell_correction_max": 3,
    }
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 7,
        "common_word_count": 3,
        "computation_time": 0.0010485649108886719,
        "domain_word_count": 0,
        "inner_product": -21 + 2.1,
        "innovation_word_count": 0,
        "processed_response": " ".join(["response"] * 3)
        + " "
        + " ".join(["nonsense_word"] * 7),
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": " ".join(["respones"] * 10),
        "spelling_correction": "auto",
        "spelling_correction_used": True,
        "num_spelling_correction": 3,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": False,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_auto_spelling_correction_limit_10(client):
    """Various numerics"""

    params = {
        "response": " ".join(["respones"] * 10),
        "spelling_correction": "auto",
        "spell_correction_max": 10,
    }
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 10,
        "computation_time": 0.0010485649108886719,
        "domain_word_count": 0,
        "inner_product": 7,
        "innovation_word_count": 0,
        "processed_response": " ".join(["response"] * 10),
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": " ".join(["respones"] * 10),
        "spelling_correction": "auto",
        "spelling_correction_used": True,
        "num_spelling_correction": 10,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_spelling_correction_default(client):
    """Various numerics"""

    params = {"response": "This is my respones", "spelling_correction": "odd"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 1,
        "computation_time": 0.0010485649108886719,
        "domain_word_count": 0,
        "inner_product": 0.7,
        "innovation_word_count": 0,
        "processed_response": "response",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "This is my respones",
        "spelling_correction": PARSER_DEFAULTS["spelling_correction"],
        "spelling_correction_used": True,
        "num_spelling_correction": 1,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": False,
        "uid_used": None,
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_stem_option_words(client):
    """Word in the domain of the exercise (the book)"""

    params = {"response": "example leg", "uid": "9@6"}
    params.update(QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 0,
        "computation_time": 0.004462242126464844,
        "domain_word_count": 0,
        "inner_product": 2.0,
        "innovation_word_count": 0,
        "intercept": 1,
        "lazy_math_evaluation": True,
        "num_spelling_correction": 0,
        "option_word_count": 1,
        "processed_response": "example leg",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "example leg",
        "spelling_correction": "auto",
        "spelling_correction_used": False,
        "stem_word_count": 1,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": True,
        "uid_used": "9@7",
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()


def test_no_stem_option_words(client):
    """Word in the domain of the exercise (the book)"""

    params = {"response": "example leg", "uid": "9@6"}
    params.update(NO_QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 2,
        "computation_time": 0.005262136459350586,
        "domain_word_count": 0,
        "inner_product": 1.4,
        "innovation_word_count": 0,
        "intercept": 1,
        "lazy_math_evaluation": True,
        "num_spelling_correction": 0,
        "option_word_count": 0,
        "processed_response": "example leg",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "example leg",
        "spelling_correction": "auto",
        "spelling_correction_used": False,
        "stem_word_count": 0,
        "tag_numeric": True,
        "tag_numeric_input": "auto",
        "uid_found": True,
        "uid_used": "9@7",
        "valid": True,
        "version": "testing",
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert "." in result["version"]
    result["version"] = "testing"
    assert expected.items() <= result.items()

def test_tag_numeric_no_question(client):
    """With no question supplied, tag_numeric should be True"""

    params = {"response": "1.0 5 10 some words"}
    params.update(QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    result = resp.json
    assert result["tag_numeric"] == True
    assert result["valid"] == True

def test_tag_numeric_auto_no_question(client):
    """With no question supplied, tag_numeric should be True"""

    params = {"response": "1.0 5 10 some words", "tag_numeric": "auto"}
    params.update(QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    result = resp.json
    assert result["tag_numeric"] == True

def test_tag_numeric_auto_numeric_question(client):
    """With no question supplied, tag_numeric should be True"""

    params = {"response": "1.0 5 some words", "tag_numeric": "auto", "uid": "9@6"}
    params.update(QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    result = resp.json
    assert result["tag_numeric"] == True
    assert result["valid"] == True

def test_tag_numeric_auto_nonnumeric_question(client):
    """With no question supplied, tag_numeric should be True"""

    params = {"response": "1.0 5 10 some words", "tag_numeric": "auto", "uid": "100@1"}
    params.update(QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    result = resp.json
    assert result["tag_numeric"] == False
    assert result["valid"] == False

def test_tag_numeric_true_nonnumeric_question(client):
    """With no question supplied, tag_numeric should be True"""

    params = {"response": "1.0 5 10 some words", "tag_numeric": True, "uid": "100@1"}
    params.update(QUESTION_WEIGHT_DICT)
    resp = client.get("/validate", query_string=urlencode(params))
    result = resp.json
    assert result["tag_numeric"] == True
    assert result["valid"] == True
