import copy
import pytest
import os
import shutil
import tempfile
import time


import vcr

from validator import app, __version__ as app_version

start_time = time.ctime()


@pytest.fixture(scope="module")
def test_app():
    tmpdir = tempfile.mkdtemp()
    write_app = app.create_app(DATA_DIR=tmpdir)
    yield write_app
    shutil.rmtree(tmpdir, ignore_errors=True)

@pytest.fixture(scope="module")
def client(test_app):
    test_app.config["TESTING"] = True
    client = test_app.test_client()
    yield client

@pytest.fixture(scope="module")
def test_app_with_data():
    tmpdir = tempfile.mkdtemp()
    write_app = app.create_app(DATA_DIR=tmpdir)
    yield write_app
    for filename in os.listdir('tests/data'):
        shutil.copy(os.path.join('tests/data', filename), tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)

@pytest.fixture(scope="module")
def client_with_data(test_app_with_data):
    test_app_with_data.config["TESTING"] = True
    client_with_data = test_app_with_data.test_client()
    yield client_with_data

@pytest.fixture(scope="module")
def import_yaml(test_app_with_data, client_with_data):
    data_dir = test_app_with_data.config["DATA_DIR"]
    if os.listdir(data_dir) != []:
        raise LookupError(f"Error! pointing at existing data files at {data_dir}")
    with vcr.use_cassette("tests/cassettes/import.yaml"):
        res = client_with_data.post(
            "/import",
            data={
                "file": (
                    "tutor_manifests/"
                    "Introduction_to_Sociology_2e_02040312-72c8-441e-a685-20e9333f3e1d_10.1.yml",
                    "test.yml",
                )
            },
        )
    yield res

    data_dir = test_app_with_data.config["DATA_DIR"]
    data_files = os.listdir(data_dir)
    for dfile in data_files:
        os.remove(os.path.join(data_dir, dfile))


EXPECTED_BOOK_NAMES = set(["Introduction to Sociology 2e"])

EXPECTED_VOCABULARIES = ["domain", "innovation", "questions"]


EXPECTED_FEATURE_WEIGHTS = {
  "default_id":"d3732be6-a759-43aa-9e1a-3e9bd94f8b6b",
  "d3732be6-a759-43aa-9e1a-3e9bd94f8b6b": {
    "stem_word_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7
  },
  "cc2ed0ea-46cc-428f-b8e4-136df5b157db": {
    "stem_word_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7
  },
  "566ceadc-3835-4b08-9dea-ac6fcbb27c96": {
    "stem_word_count": 1,
    "option_word_count": 1,
    "innovation_word_count": 0,
    "domain_word_count": 0,
    "bad_word_count": -3,
    "common_word_count": 0.7
  },
  "f84e554a-c06c-11ea-a880-7f87cd92d175": {}
}


BOOK_VUID = "02040312-72c8-441e-a685-20e9333f3e1d@10.1"
BOOK_NAME = "Introduction to Sociology 2e"
NUM_PAGES = 96
NUM_DOMAIN_WORDS = 7592
INNOVATION_PAGE_VUID = "325e4afd-80b6-44dd-87b6-35aff4f40eac@6"
NUM_PAGE_INNOVATION_WORDS = 199
NUM_PAGES_WITH_QUESTIONS = 82
QUESTION_PAGE_VUID = "08e4a1f1-738c-4296-b07d-e13fa2973681@3"
NUM_PAGE_QUESTIONS = 20
EXERCISE_UID = "6012@2"
NUM_EXERCISE_OPTION_WORDS = 30
NUM_EXERCISE_STEM_WORDS = 6

NOT_BOOK_VUID = "67be4044-bf7f-4b50-8798-bcd8a88ca5b6@1"

DEFAULT_FEATURE_WEIGHTS_ID = "d3732be6-a759-43aa-9e1a-3e9bd94f8b6b"

NEW_DEFAULT_ID = "cc2ed0ea-46cc-428f-b8e4-136df5b157db"

EMPTY_FEATURE_WEIGHTS = {}

DEFAULT_FEATURE_WEIGHTS = {
    "stem_word_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7,
    "intercept": 0,
}

NEW_FEATURE_WEIGHTS = {
    "stem_word_count": 0.4,
    "option_word_count": 0.1,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7,
    "intercept": 0,
}

INCOMPLETE_FEATURE_WEIGHTS = {
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7,
    "intercept": 0,
}


EXTRA_FEATURE_WEIGHTS = {
    "stem_word_count": 0,
    "nonsense_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7,
    "intercept": 0,
}


def test_status(client_with_data, import_yaml):
    """Status reports loaded books, version, and start time"""

    resp = client_with_data.get("/status")
    json_status = resp.json

    assert json_status["started"][:11] == start_time[:11]
    assert json_status["started"][19:] == start_time[19:]

    assert json_status["version"]["version"] == app_version

    assert set(json_status["datasets"].keys()) == set(["books", "feature_weights"])

    assert set(json_status["datasets"]["books"][0].keys()) == set(["name", "vuid"])

    returned_book_names = set([b["name"] for b in json_status["datasets"]["books"]])

    assert EXPECTED_BOOK_NAMES == returned_book_names

    assert json_status["datasets"]["feature_weights"] == list(
        EXPECTED_FEATURE_WEIGHTS.keys()
    )


def test_datasets_books(client_with_data, import_yaml):
    """List of available books"""

    resp = client_with_data.get("/datasets/books")
    json_status = resp.json

    returned_book_names = set([b["name"] for b in json_status])

    assert EXPECTED_BOOK_NAMES == returned_book_names

    assert json_status[0]["vocabularies"] == EXPECTED_VOCABULARIES


def test_books_book(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}")
    assert resp.status_code == 200
    assert resp.json["name"] == BOOK_NAME
    assert resp.json["vuid"] == BOOK_VUID
    assert len(resp.json["pages"]) == NUM_PAGES
    assert resp.json["vocabularies"] == EXPECTED_VOCABULARIES


def test_book_vocabularies(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/vocabularies")
    assert resp.status_code == 200
    assert resp.json == EXPECTED_VOCABULARIES


def test_book_vocab_bad(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/vocabularies/nosuchvocab")
    assert resp.status_code == 404


def test_book_vocab_domain(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/vocabularies/domain")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_DOMAIN_WORDS
    assert "anecdote" in resp.json


def test_book_vocab_innovation(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/vocabularies/innovation")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES
    page = resp.json[0]
    assert set(page.keys()) == set(["innovation_words", "page_vuid"])
    assert page["page_vuid"] == INNOVATION_PAGE_VUID
    assert len(page["innovation_words"]) == NUM_PAGE_INNOVATION_WORDS
    assert "relevance" in page["innovation_words"]


def test_book_vocab_page_innovation(client_with_data, import_yaml):
    resp = client_with_data.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/innovation/{INNOVATION_PAGE_VUID}"
    )
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGE_INNOVATION_WORDS
    assert "relevance" in resp.json


def test_book_vocab_questions(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/vocabularies/questions")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES_WITH_QUESTIONS
    page = resp.json[0]
    assert set(page.keys()) == set(["questions", "page_vuid"])
    assert page["page_vuid"] == QUESTION_PAGE_VUID
    assert len(page["questions"]) == NUM_PAGE_QUESTIONS
    question = page["questions"][0]
    assert question["exercise_uid"] == EXERCISE_UID
    assert len(question["option_words"]) == NUM_EXERCISE_OPTION_WORDS
    assert len(question["stem_words"]) == NUM_EXERCISE_STEM_WORDS
    assert "sociological" in question["stem_words"]
    assert "extroverts" in question["option_words"]


def test_book_vocab_page_questions(client_with_data, import_yaml):
    resp = client_with_data.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/questions/{QUESTION_PAGE_VUID}"
    )
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGE_QUESTIONS
    question = resp.json[0]
    assert question["exercise_uid"] == EXERCISE_UID
    assert len(question["option_words"]) == NUM_EXERCISE_OPTION_WORDS
    assert len(question["stem_words"]) == NUM_EXERCISE_STEM_WORDS
    assert "sociological" in question["stem_words"]
    assert "extroverts" in question["option_words"]


def test_book_vocab_page_questions_no_questions(client_with_data, import_yaml):
    resp = client_with_data.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/questions/{INNOVATION_PAGE_VUID}"
    )
    assert resp.status_code == 200
    assert resp.json == []


def test_book_vocab_page_questions_not_in_book(client_with_data, import_yaml):
    resp = client_with_data.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/questions/{NOT_BOOK_VUID}"
    )
    assert resp.status_code == 404
    assert resp.json["message"] == "No such page in book"


def test_book_pages(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/pages")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES


def test_book_page(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/books/{BOOK_VUID}/pages/{INNOVATION_PAGE_VUID}")
    assert resp.status_code == 200


NUM_QUESTIONS = 1485
NUM_QUESTIONS_UID = 1


def test_datasets_questions(client_with_data, import_yaml):
    resp = client_with_data.get("/datasets/questions")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_QUESTIONS


def test_datasets_questions_uid(client_with_data, import_yaml):
    resp = client_with_data.get(f"/datasets/questions/{EXERCISE_UID}")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_QUESTIONS_UID
    question = resp.json[0]
    assert question["exercise_uid"] == EXERCISE_UID
    assert len(question["option_words"]) == NUM_EXERCISE_OPTION_WORDS
    assert len(question["stem_words"]) == NUM_EXERCISE_STEM_WORDS
    assert "sociological" in question["stem_words"]
    assert "extroverts" in question["option_words"]


def test_empty_feature_weights(client_with_data):
    resp = client_with_data.post("/datasets/feature_weights", json=EMPTY_FEATURE_WEIGHTS)
    assert resp.status_code == 400
    assert resp.json["message"] == "Incomplete or incorrect feature weight keys"


def test_incomplete_feature_weights(client_with_data):
    resp = client_with_data.post("/datasets/feature_weights", json=INCOMPLETE_FEATURE_WEIGHTS)
    assert resp.status_code == 400
    assert resp.json["message"] == "Incomplete or incorrect feature weight keys"


def test_extra_feature_weights(client_with_data):
    resp = client_with_data.post("/datasets/feature_weights", json=EXTRA_FEATURE_WEIGHTS)
    assert resp.status_code == 400
    assert resp.json["message"] == "Incomplete or incorrect feature weight keys"


def test_default_feature_weights(client_with_data):
    resp = client_with_data.post("/datasets/feature_weights", json=DEFAULT_FEATURE_WEIGHTS)
    assert resp.status_code == 200
    assert resp.json == {
        "msg": "Feature weights successfully imported.",
        "feature_weight_set_id": DEFAULT_FEATURE_WEIGHTS_ID,
    }
    resp = client_with_data.get(f"/datasets/feature_weights/{DEFAULT_FEATURE_WEIGHTS_ID}")
    assert resp.json == DEFAULT_FEATURE_WEIGHTS


def test_new_feature_weights(client_with_data):
    resp = client_with_data.post("/datasets/feature_weights", json=NEW_FEATURE_WEIGHTS)
    assert resp.status_code == 200
    assert resp.json["msg"] == "Feature weights successfully imported."
    new_feature_weights_id = resp.json["feature_weight_set_id"]
    assert (client_with_data.get(f"/datasets/feature_weights/{new_feature_weights_id}")).json == NEW_FEATURE_WEIGHTS

    second_app = app.create_app(DATA_DIR=client_with_data.application.config["DATA_DIR"])
    second_app.config["TESTING"] = True
    second_client = second_app.test_client()
    assert (second_client.get(f"/datasets/feature_weights/{new_feature_weights_id}")).json == NEW_FEATURE_WEIGHTS


def test_set_default_feature_weights(client_with_data):
    resp = client_with_data.put("/datasets/feature_weights/default", json=NEW_DEFAULT_ID)
    assert resp.status_code == 200
    assert resp.json["msg"] == "Successfully set default feature weight id."
    assert (client_with_data.get(f"/datasets/feature_weights/default")).json == NEW_DEFAULT_ID

    second_app = app.create_app(DATA_DIR=client.application.config["DATA_DIR"])
    second_app.config["TESTING"] = True
    second_client = second_app.test_client()
    assert (second_client.get(f"/datasets/feature_weights/default")).json == NEW_DEFAULT_ID
