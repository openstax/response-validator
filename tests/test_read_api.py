import time
import pytest

from flask import current_app
from validator import app, __version__ as app_version

myapp = app.create_app(DATA_DIR="tests/data")

start_time = time.ctime()


@pytest.fixture(scope="module")
def client():
    myapp.config["TESTING"] = True
    client = myapp.test_client()
    yield client


def test_404(client):
    """Top level does nothing."""

    resp = client.get("/")
    assert resp.status_code == 404


def test_ping(client):
    """ping -> pong"""

    resp = client.get("/ping")
    assert resp.data.decode(resp.content_encoding or "utf-8") == "pong"


def test_version(client):
    """Simple version string"""

    resp = client.get("/version")
    assert resp.data.decode(resp.content_encoding or "utf-8") == app_version


EXPECTED_BOOK_NAMES = set(
    [
        "Introduction to Sociology 2e",
        "Biology for AP® Courses",
        "Biology 2e",
        "College Physics with Courseware",
        "College Physics for AP® Courses",
    ]
)

EXPECTED_VOCABULARIES = ["domain", "innovation", "questions"]
EXPECTED_FEATURE_WEIGHTS = {
  "default_id": "d3732be6-a759-43aa-9e1a-3e9bd94f8b6b",
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

expected_fw_ids = list(EXPECTED_FEATURE_WEIGHTS.keys())
expected_fw_ids.remove("default_id")


DEFAULT_FEATURE_WEIGHTS_SET = {
    "stem_word_count": 0,
    "option_word_count": 0,
    "innovation_word_count": 2.2,
    "domain_word_count": 2.5,
    "bad_word_count": -3,
    "common_word_count": 0.7,
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
NUM_EXERCISE_OPTION_WORDS = 26
NUM_EXERCISE_STEM_WORDS = 6

NOT_BOOK_VUID = "67be4044-bf7f-4b50-8798-bcd8a88ca5b6@1"

DEFAULT_FEATURE_WEIGHT_ID = "d3732be6-a759-43aa-9e1a-3e9bd94f8b6b"

NOT_FEATURE_WEIGHT_ID = "67be4044-bf7f-4b50-8798-bcd8a88ca5b6"


def test_status(client):
    """Status reports loaded books, version, and start time"""

    resp = client.get("/status")
    json_status = resp.json

    assert json_status["started"][:11] == start_time[:11]
    assert json_status["started"][19:] == start_time[19:]

    assert json_status["version"]["version"] == app_version

    assert set(json_status["datasets"].keys()) == set(["books", "feature_weights"])
    assert set(json_status["datasets"]["books"][0].keys()) == set(["feature_weights_id", "name", "vuid"])

    returned_book_names = set([b["name"] for b in json_status["datasets"]["books"]])

    assert EXPECTED_BOOK_NAMES == returned_book_names
    # add code that checks fw

    assert json_status["datasets"]["feature_weights"] == expected_fw_ids


def test_fetch_feature_weights_ids(client):
    resp = client.get("/datasets/feature_weights")
    assert resp.json == expected_fw_ids


def test_fetch_default_feature_weights_id(client):
    resp = client.get("/status/defaults/feature_weights_id")
    assert resp.json == DEFAULT_FEATURE_WEIGHT_ID


def test_fetch_default_feature_weights_set(client):
    resp = client.get("/status/defaults")
    assert resp.json == DEFAULT_FEATURE_WEIGHTS_SET


def test_datasets(client):
    """List of available datasets"""

    resp = client.get("/datasets")
    assert resp.json == ["books", "questions", "feature_weights"]


def test_datasets_books(client):
    """List of available books"""

    resp = client.get("/datasets/books")
    json_status = resp.json

    returned_book_names = set([b["name"] for b in json_status])

    assert EXPECTED_BOOK_NAMES == returned_book_names

    assert json_status[0]["vocabularies"] == EXPECTED_VOCABULARIES


def test_books_bad_no_version(client):
    resp = client.get("/datasets/books/nosuchbook")
    assert resp.status_code == 400
    assert resp.json["message"] == "Need uuid and version"


def test_books_bad_vuid(client):
    resp = client.get("/datasets/books/nosuchbook@4")
    assert resp.status_code == 400
    assert resp.json["message"] == "Not a valid uuid for book"


def test_books_bad_version(client):
    NOT_BOOK_UUID = NOT_BOOK_VUID[:-2]
    resp = client.get(f"/datasets/books/{NOT_BOOK_UUID}@draft")
    assert resp.status_code == 400
    assert resp.json["message"] == "Bad version"


def test_books_bad_not_found(client):
    resp = client.get(f"/datasets/books/{NOT_BOOK_VUID}")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_books_book(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}")
    assert resp.status_code == 200
    assert resp.json["name"] == BOOK_NAME
    assert resp.json["vuid"] == BOOK_VUID
    assert resp.json["feature_weights_id"] == ''
    assert len(resp.json["pages"]) == NUM_PAGES
    assert resp.json["vocabularies"] == EXPECTED_VOCABULARIES


def test_book_vocabularies(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/vocabularies")
    assert resp.status_code == 200
    assert resp.json == EXPECTED_VOCABULARIES


def test_book_vocab_bad(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/vocabularies/nosuchvocab")
    assert resp.status_code == 404


def test_book_vocab_domain_not_found(client):
    resp = client.get(f"/datasets/books/{NOT_BOOK_VUID}/vocabularies/domain")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_domain(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/vocabularies/domain")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_DOMAIN_WORDS
    assert "anecdote" in resp.json


def test_book_vocab_innovation(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/vocabularies/innovation")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES
    page = resp.json[0]
    assert set(page.keys()) == set(["innovation_words", "page_vuid"])
    assert page["page_vuid"] == INNOVATION_PAGE_VUID
    assert len(page["innovation_words"]) == NUM_PAGE_INNOVATION_WORDS
    assert "relevance" in page["innovation_words"]


def test_book_vocab_innovation_not_found(client):
    resp = client.get(f"/datasets/books/{NOT_BOOK_VUID}/vocabularies/innovation")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_page_innovation(client):
    resp = client.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/innovation/{INNOVATION_PAGE_VUID}"
    )
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGE_INNOVATION_WORDS
    assert "relevance" in resp.json


def test_book_vocab_page_innovation_not_found(client):
    resp = client.get(
        f"/datasets/books/{NOT_BOOK_VUID}/vocabularies/innovation/{INNOVATION_PAGE_VUID}"
    )
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book or page"


def test_book_vocab_questions(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/vocabularies/questions")
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


def test_book_vocab_questions_not_found(client):
    resp = client.get(f"/datasets/books/{NOT_BOOK_VUID}/vocabularies/questions")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_page_questions(client):
    resp = client.get(
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


def test_book_vocab_page_questions_not_found(client):
    resp = client.get(
        f"/datasets/books/{NOT_BOOK_VUID}/vocabularies/questions/{QUESTION_PAGE_VUID}"
    )
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_page_questions_no_questions(client):
    resp = client.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/questions/{INNOVATION_PAGE_VUID}"
    )
    assert resp.status_code == 200
    assert resp.json == []


def test_book_vocab_page_questions_not_in_book(client):
    resp = client.get(
        f"/datasets/books/{BOOK_VUID}/vocabularies/questions/{NOT_BOOK_VUID}"
    )
    assert resp.status_code == 404
    assert resp.json["message"] == "No such page in book"


def test_book_pages(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/pages")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES


def test_book_pages_no_book(client):
    resp = client.get(f"/datasets/books/{NOT_BOOK_VUID}/pages")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_page(client):
    resp = client.get(f"/datasets/books/{BOOK_VUID}/pages/{INNOVATION_PAGE_VUID}")
    assert resp.status_code == 200


def test_book_page_no_book(client):
    resp = client.get(f"/datasets/books/{NOT_BOOK_VUID}/pages/{INNOVATION_PAGE_VUID}")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book or page"


NUM_QUESTIONS = 23218
NUM_QUESTIONS_UID = 1


def test_datasets_questions(client):
    resp = client.get("/datasets/questions")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_QUESTIONS


def test_datasets_questions_uid(client):
    resp = client.get(f"/datasets/questions/{EXERCISE_UID}")
    assert resp.status_code == 200
    assert len(resp.json) == NUM_QUESTIONS_UID
    question = resp.json[0]
    assert question["exercise_uid"] == EXERCISE_UID
    assert len(question["option_words"]) == NUM_EXERCISE_OPTION_WORDS
    assert len(question["stem_words"]) == NUM_EXERCISE_STEM_WORDS
    assert "sociological" in question["stem_words"]
    assert "extroverts" in question["option_words"]


def test_feature_weights_bad_uuid(client):
    resp = client.get("/datasets/feature_weights/nosuchfw@4")
    assert resp.status_code == 400
    assert resp.json["message"] == "Not a valid uuid for feature weights"


def test_feature_weights_bad_not_found(client):
    resp = client.get(f"/datasets/feature_weights/{NOT_FEATURE_WEIGHT_ID}")
    assert resp.status_code == 404
    assert resp.json["message"] == "No such set of feature weights"


def test_dataset_feature_weights(client):
    resp = client.get(f"/datasets/feature_weights/{DEFAULT_FEATURE_WEIGHT_ID}")
    assert resp.status_code == 200
    assert resp.json == EXPECTED_FEATURE_WEIGHTS[DEFAULT_FEATURE_WEIGHT_ID]


def test_dataset_default_feature_weights(client):
    resp = client.get("/datasets/feature_weights/default")
    assert resp.status_code == 200
    assert resp.json == client.application.datasets["feature_weights"]["default_id"]
