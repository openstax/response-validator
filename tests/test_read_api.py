import time
import pytest

from validator import app

start_time = time.ctime()


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    client = app.app.test_client()
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
    assert resp.data.decode(resp.content_encoding or "utf-8") == app.__version__


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

BOOK_VUID = '02040312-72c8-441e-a685-20e9333f3e1d@10.1'
BOOK_NAME = "Introduction to Sociology 2e"
NUM_PAGES = 96
NUM_DOMAIN_WORDS = 7592
INNOVATION_PAGE_VUID = '325e4afd-80b6-44dd-87b6-35aff4f40eac@6'
NUM_PAGE_INNOVATION_WORDS = 199
NUM_PAGES_WITH_QUESTIONS = 82
QUESTION_PAGE_VUID = '08e4a1f1-738c-4296-b07d-e13fa2973681@3'
NUM_PAGE_QUESTIONS = 20
EXERCISE_UID = '6012@2'
NUM_EXERCISE_OPTION_WORDS = 26
NUM_EXERCISE_STEM_WORDS = 6

NOT_BOOK_VUID = "67be4044-bf7f-4b50-8798-bcd8a88ca5b6@1"


def test_status(client):
    """Status reports loaded books, version, and start time"""

    resp = client.get("/status")
    json_status = resp.json

    assert json_status["started"][:11] == start_time[:11]
    assert json_status["started"][19:] == start_time[19:]

    assert json_status["version"]["version"] == app.__version__

    assert set(json_status["datasets"].keys()) == set(["books"])
    assert set(json_status["datasets"]["books"][0].keys()) == set(
        ["name", "vuid"]
    )

    returned_book_names = set([b["name"] for b in json_status["datasets"]["books"]])

    assert EXPECTED_BOOK_NAMES == returned_book_names


def test_datasets(client):
    """List of available datasets"""

    resp = client.get("/datasets")
    assert resp.json == ["books", "questions"]


def test_datasets_books(client):
    """List of available books"""

    resp = client.get("/datasets/books")
    json_status = resp.json

    returned_book_names = set([b["name"] for b in json_status])

    assert EXPECTED_BOOK_NAMES == returned_book_names

    assert json_status[0]["vocabularies"] == EXPECTED_VOCABULARIES


def test_books_bad_no_version(client):
    resp = client.get('/datasets/books/nosuchbook')
    assert resp.status_code == 400
    assert resp.json["message"] == "Need uuid and version"


def test_books_bad_vuid(client):
    resp = client.get('/datasets/books/nosuchbook@4')
    assert resp.status_code == 400
    assert resp.json["message"] == "Not a valid uuid for book"


def test_books_bad_version(client):
    NOT_BOOK_UUID = NOT_BOOK_VUID[:-2]
    resp = client.get(f'/datasets/books/{NOT_BOOK_UUID}@draft')
    assert resp.status_code == 400
    assert resp.json["message"] == "Bad version"


def test_books_bad_not_found(client):
    resp = client.get(f'/datasets/books/{NOT_BOOK_VUID}')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_books_book(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}')
    assert resp.status_code == 200
    assert resp.json["name"] == BOOK_NAME
    assert resp.json["vuid"] == BOOK_VUID
    assert len(resp.json["pages"]) == NUM_PAGES
    assert resp.json["vocabularies"] == EXPECTED_VOCABULARIES


def test_book_vocabularies(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies')
    assert resp.status_code == 200
    assert resp.json == EXPECTED_VOCABULARIES


def test_book_vocab_bad(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/nosuchvocab')
    assert resp.status_code == 404


def test_book_vocab_domain_not_found(client):
    resp = client.get(f'/datasets/books/{NOT_BOOK_VUID}/vocabularies/domain')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_domain(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/domain')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_DOMAIN_WORDS
    assert 'anecdote' in resp.json


def test_book_vocab_innovation(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/innovation')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES
    page = resp.json[0]
    assert set(page.keys()) == set(['innovation_words', 'page_vuid'])
    assert page["page_vuid"] == INNOVATION_PAGE_VUID
    assert len(page["innovation_words"]) == NUM_PAGE_INNOVATION_WORDS
    assert 'relevance' in page['innovation_words']


def test_book_vocab_innovation_not_found(client):
    resp = client.get(f'/datasets/books/{NOT_BOOK_VUID}/vocabularies/innovation')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_page_innovation(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/innovation/{INNOVATION_PAGE_VUID}')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGE_INNOVATION_WORDS
    assert 'relevance' in resp.json


def test_book_vocab_page_innovation_not_found(client):
    resp = client.get(
        f'/datasets/books/{NOT_BOOK_VUID}/vocabularies/innovation/{INNOVATION_PAGE_VUID}')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book or page"


def test_book_vocab_questions(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/questions')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES_WITH_QUESTIONS
    page = resp.json[0]
    assert set(page.keys()) == set(['questions', 'page_vuid'])
    assert page["page_vuid"] == QUESTION_PAGE_VUID
    assert len(page["questions"]) == NUM_PAGE_QUESTIONS
    question = page["questions"][0]
    assert question['exercise_uid'] == EXERCISE_UID
    assert len(question['option_words']) == NUM_EXERCISE_OPTION_WORDS
    assert len(question['stem_words']) == NUM_EXERCISE_STEM_WORDS
    assert 'sociological' in question['stem_words']
    assert 'extroverts' in question['option_words']


def test_book_vocab_questions_not_found(client):
    resp = client.get(f'/datasets/books/{NOT_BOOK_VUID}/vocabularies/questions')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_page_questions(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/questions/{QUESTION_PAGE_VUID}')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGE_QUESTIONS
    question = resp.json[0]
    assert question['exercise_uid'] == EXERCISE_UID
    assert len(question['option_words']) == NUM_EXERCISE_OPTION_WORDS
    assert len(question['stem_words']) == NUM_EXERCISE_STEM_WORDS
    assert 'sociological' in question['stem_words']
    assert 'extroverts' in question['option_words']


def test_book_vocab_page_questions_not_found(client):
    resp = client.get(
        f'/datasets/books/{NOT_BOOK_VUID}/vocabularies/questions/{QUESTION_PAGE_VUID}')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_vocab_page_questions_no_questions(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/questions/{INNOVATION_PAGE_VUID}')
    assert resp.status_code == 200
    assert resp.json == []


def test_book_vocab_page_questions_not_in_book(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/vocabularies/questions/{NOT_BOOK_VUID}')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such page in book"


def test_book_pages(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/pages')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_PAGES


def test_book_pages_no_book(client):
    resp = client.get(f'/datasets/books/{NOT_BOOK_VUID}/pages')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book"


def test_book_page(client):
    resp = client.get(f'/datasets/books/{BOOK_VUID}/pages/{INNOVATION_PAGE_VUID}')
    assert resp.status_code == 200


def test_book_page_no_book(client):
    resp = client.get(f'/datasets/books/{NOT_BOOK_VUID}/pages/{INNOVATION_PAGE_VUID}')
    assert resp.status_code == 404
    assert resp.json["message"] == "No such book or page"


NUM_QUESTIONS = 19968
NUM_QUESTIONS_UID = 1


def test_datasets_questions(client):
    resp = client.get('/datasets/questions')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_QUESTIONS


def test_datasets_questions_uid(client):
    resp = client.get(f'/datasets/questions/{EXERCISE_UID}')
    assert resp.status_code == 200
    assert len(resp.json) == NUM_QUESTIONS_UID
    question = resp.json[0]
    assert question['exercise_uid'] == EXERCISE_UID
    assert len(question['option_words']) == NUM_EXERCISE_OPTION_WORDS
    assert len(question['stem_words']) == NUM_EXERCISE_STEM_WORDS
    assert 'sociological' in question['stem_words']
    assert 'extroverts' in question['option_words']