import pytest
from urllib.parse import urlencode

from validator import app


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    client = app.app.test_client()

    yield client


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
        "spelling_correction": True,
        "tag_numeric": False,
        "uid": None,
        "uid_found": False,
        "valid": False,
    }
    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
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
        "spelling_correction": True,
        "tag_numeric": False,
        "uid": None,
        "uid_found": False,
        "valid": False,
    }
    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected


def test_non_words(client):
    """Just well-known not-a-word"""

    params = {"response": "idk asdf lol n/a"}
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 4,
        "common_word_count": 0,
        "computation_time": 0.0015406608581542969,
        "domain_word_count": 0,
        "inner_product": -12.0,
        "innovation_word_count": 0,
        "processed_response": "nonsense_word nonsense_word nonsense_word n/a",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "idk asdf lol n/a",
        "spelling_correction": True,
        "tag_numeric": False,
        "uid": None,
        "uid_found": False,
        "valid": False,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected


def test_simple_words(client):
    """Just well-known not-a-word"""

    params = {"response": "Here is my response"}
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
        "spelling_correction": True,
        "tag_numeric": False,
        "uid": None,
        "uid_found": False,
        "valid": True,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected


def test_domain_words(client):
    """Word in the domain of the exercise (the book)"""

    params = {"response": "meerkats chemiosmosis", "uid": "1340@1"}
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 0,
        "computation_time": 0.07029032707214355,
        "domain_word_count": 2,
        "inner_product": 5.0,
        "innovation_word_count": 0,
        "processed_response": "meerkats chemiosmosis",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "meerkats chemiosmosis",
        "spelling_correction": True,
        "tag_numeric": False,
        "uid": "1340@1",
        "uid_found": True,
        "valid": True,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected


def test_innovation_words(client):
    """A word in the innovation list of the exercise"""

    params = {
        "response": "1.0 meerkats photographs",
        "uid": "290@1",
        "tag_numerics": True,
    }
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 0,
        "common_word_count": 0,
        "computation_time": 0.1347815990447998,
        "domain_word_count": 2,
        "inner_product": 7.2,
        "innovation_word_count": 1,
        "processed_response": "10 meerkats photographs",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "1.0 meerkats photographs",
        "spelling_correction": True,
        "tag_numeric": False,
        "uid": "290@1",
        "uid_found": True,
        "valid": True,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected


def test_numeric_words(client):
    """Various numerics"""

    params = {"response": "0 23 -3 1.2 IV", "tag_numeric": True}
    resp = client.get("/validate", query_string=urlencode(params))
    expected = {
        "bad_word_count": 1,
        "common_word_count": 4,
        "computation_time": 0.0003266334533691406,
        "domain_word_count": 0,
        "inner_product": -0.20000000000000018,
        "innovation_word_count": 0,
        "processed_response": "nonsense_word numeric_type_int numeric_type_int "
        "numeric_type_int numeric_type_roman",
        "remove_nonwords": True,
        "remove_stopwords": True,
        "response": "0 23 -3 1.2 IV",
        "spelling_correction": True,
        "tag_numeric": True,
        "uid": None,
        "uid_found": False,
        "valid": False,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected


def test_no_spelling_correction(client):
    """Various numerics"""

    params = {"response": "This is my respones", "spelling_correction": ""}
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
        "tag_numeric": False,
        "uid": None,
        "uid_found": False,
        "valid": False,
    }

    result = resp.json
    assert result["computation_time"] != 0
    expected["computation_time"] = result["computation_time"]
    assert result == expected
