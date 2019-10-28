# READ API Routes for validator
import json
import time

from flask import jsonify, Blueprint, current_app
from uuid import UUID

from . import __version__, _version

start_time = time.ctime()

bp = Blueprint("read_api", __name__, url_prefix="/")


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


@bp.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@bp.route("/datasets")
def datasets_index():
    return jsonify(["books", "questions"])  # FIXME , "feature_coefficients"])


def _books_json(include_vocabs=True):
    data = current_app.df["domain"][["book_name", "vuid"]].rename(
        {"book_name": "name"}, axis=1
    )
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


@bp.route("/datasets/books")
def books_index():
    return jsonify(_books_json())


@bp.route("/datasets/books/<vuid>")
def fetch_book(vuid):
    df = current_app.df
    data = df["domain"][df["domain"]["vuid"] == vuid][["book_name", "vuid"]].rename(
        {"book_name": "name"}, axis=1
    )
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    data["vocabularies"] = [["domain", "innovation", "questions"]] * len(data)
    page_list = (
        df["innovation"][df["innovation"]["cvuid"].str.startswith(vuid)]
        .cvuid.str.split(":", expand=True)[1]
        .tolist()
    )
    data_json = data.to_dict(orient="records")[0]
    data_json["pages"] = page_list
    return jsonify(data_json)


@bp.route("/datasets/books/<vuid>/pages")
def fetch_page_list(vuid):
    df = current_app.df
    book = df["innovation"][df["innovation"]["cvuid"].str.startswith(vuid)]
    if book.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    page_list = book.cvuid.str.split(":", expand=True)[1].tolist()
    return jsonify(page_list)


@bp.route("/datasets/books/<vuid>/pages/<pvuid>")
def fetch_page(vuid, pvuid):
    df = current_app.df
    innovation = df["innovation"][df["innovation"]["cvuid"] == ":".join((vuid, pvuid))][
        "innovation_words"
    ]
    if innovation.empty:
        _validate_vuid(vuid)
        _validate_vuid(pvuid, vuid_type="page")
        raise InvalidUsage("No such book or page", status_code=404)

    questions = (
        df["questions"][df["questions"]["cvuid"] == ":".join((vuid, pvuid))][
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


@bp.route("/datasets/books/<vuid>/vocabularies")
def fetch_vocabs(vuid):
    return jsonify(["domain", "innovation", "questions"])


@bp.route("/datasets/books/<vuid>/vocabularies/domain")
def fetch_domain(vuid):
    df = current_app.df
    data = df["domain"][df["domain"]["vuid"] == vuid]["domain_words"]
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    return jsonify(list(data.tolist()[0]))


@bp.route("/datasets/books/<vuid>/vocabularies/innovation")
def fetch_innovation(vuid):
    df = current_app.df
    data = df["innovation"][df["innovation"]["cvuid"].str.startswith(vuid)][
        ["cvuid", "innovation_words"]
    ]
    if data.empty:
        _validate_vuid(vuid)
        raise InvalidUsage("No such book", status_code=404)

    data["page_vuid"] = data.cvuid.str.split(":", expand=True)[1]
    data["innovation_words"] = data["innovation_words"].map(list)
    return jsonify(data[["page_vuid", "innovation_words"]].to_dict(orient="records"))


@bp.route("/datasets/books/<vuid>/vocabularies/innovation/<pvuid>")
def fetch_page_innovation(vuid, pvuid):
    df = current_app.df
    data = df["innovation"][df["innovation"]["cvuid"] == ":".join((vuid, pvuid))][
        "innovation_words"
    ]
    if data.empty:
        _validate_vuid(vuid)
        _validate_vuid(pvuid, vuid_type="page")
        raise InvalidUsage("No such book or page", status_code=404)

    return jsonify(list(data.iloc[0]))


@bp.route("/datasets/books/<vuid>/vocabularies/questions")
def fetch_questions(vuid):
    df = current_app.df
    data = df["questions"][df["questions"]["cvuid"].str.startswith(vuid)].rename(
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


@bp.route("/datasets/books/<vuid>/vocabularies/questions/<pvuid>")
def fetch_page_questions(vuid, pvuid):
    df = current_app.df
    data = df["questions"][df["questions"]["cvuid"] == ":".join((vuid, pvuid))].rename(
        {"uid": "exercise_uid", "mc_words": "option_words"}, axis=1
    )
    if data.empty:
        _validate_vuid(vuid)
        _validate_vuid(pvuid, vuid_type="page")
        book = df["domain"][df["domain"]["vuid"] == vuid]
        if book.empty:
            raise InvalidUsage("No such book", status_code=404)
        page = df["innovation"][df["innovation"]["cvuid"] == ":".join((vuid, pvuid))]
        if page.empty:
            raise InvalidUsage("No such page in book", status_code=404)

        return jsonify([])

    json_data = json.loads(
        data[["exercise_uid", "option_words", "stem_words"]].to_json(orient="records")
    )
    return jsonify(json_data)


@bp.route("/datasets/questions")
def questions_index():
    return jsonify(current_app.df["questions"].uid.tolist())


@bp.route("/datasets/questions/<uid>")
def fetch_question(uid):
    df = current_app.df
    data = df["questions"][df["questions"]["uid"] == uid].rename(
        {"uid": "exercise_uid", "mc_words": "option_words"}, axis=1
    )

    json_data = json.loads(
        data[["exercise_uid", "option_words", "stem_words"]].to_json(orient="records")
    )
    return jsonify(json_data)


@bp.route("/ping")
def ping():
    return "pong"


@bp.route("/status")
def status():
    global start_time
    data = {"version": _version.get_versions(), "started": start_time}

    if "vuid" in current_app.df["domain"].columns:
        data["datasets"] = {"books": _books_json(include_vocabs=False)}

    return jsonify(data)


@bp.route("/version")
@bp.route("/rev.txt")
def simple_version():
    return __version__
