# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import jsonify, request, Blueprint, current_app
from flask_cors import cross_origin

import pkg_resources
import uuid

from .ecosystem_importer import EcosystemImporter
from .utils import write_fixed_data, write_feature_weights
from .read_api import InvalidUsage, handle_invalid_usage


CORPORA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")

bp = Blueprint("write_api", __name__, url_prefix="/")

bp.register_error_handler(InvalidUsage, handle_invalid_usage)


# Instantiate the ecosystem importer that will be used by the import route
ecosystem_importer = EcosystemImporter(
    common_vocabulary_filename=f"{CORPORA_PATH}/big.txt"
)


def update_fixed_data(df_domain_, df_innovation_, df_questions_):

    # AEW: I feel like I am sinning against nature here . . .
    # Do we need to store these in a Redis cache or db???
    # This was all well and good before we ever tried to modify things
    df = current_app.df

    # Remove any entries from the domain, innovation, and question dataframes
    # that are duplicated by the new data
    book_id = df_domain_.iloc[0]["vuid"]
    if "vuid" in df["domain"].columns:
        df["domain"] = df["domain"][df["domain"]["vuid"] != book_id]
    if "cvuid" in df["domain"].columns:
        df["innovation"] = df["innovation"][
            ~(df["innovation"]["cvuid"].star.startswith(book_id))
        ]
    uids = df_questions_["uid"].unique()
    if "uid" in df["questions"].columns:
        df["questions"] = df["questions"][
            ~(
                df["questions"]["uid"].isin(uids)
                & df["questions"]["cvuid"].str.startswith(book_id)
            )
        ]

    # Now append the new dataframes to the in-memory ones
    df["domain"] = df["domain"].append(df_domain_, sort=False)
    df["innovation"] = df["innovation"].append(df_innovation_, sort=False)
    df["questions"] = df["questions"].append(df_questions_, sort=False)

    # Update qid sets - for shortcutting question lookup
    for idcol in ("uid", "qid"):
        current_app.qids[idcol] = set(df["questions"][idcol].values.tolist())

    # Finally, write the updated dataframes to disk and declare victory
    data_dir = current_app.config["DATA_DIR"]
    write_fixed_data(df["domain"], df["innovation"], df["questions"], data_dir)


def store_feature_weights(new_feature_weights):
    # Allows removing duplicate sets in feature weights
    # Sees if the incoming set matches with fw set

    df = current_app.df
    for fw_id, existing_feature_weights in df["feature_weights"].items():

        if existing_feature_weights == new_feature_weights:
            result_id = fw_id
            break
    else:
        result_id = uuid.uuid4()
        df["feature_weights"][str(result_id)] = new_feature_weights
        data_dir = current_app.config["DATA_DIR"]
        write_feature_weights(df["feature_weights"], data_dir)

    return result_id


def write_default_feature_weights_id(new_default_id):
    # Allows removing duplicate sets in feature weights
    # Sees if the incoming set matches with fw set

    df = current_app.df

    if new_default_id == df["feature_weights"]["default_id"]:
        return new_default_id

    else:
        df["feature_weights"]["default_id"] = new_default_id
        data_dir = current_app.config["DATA_DIR"]
        write_feature_weights(df["feature_weights"], data_dir)

    return new_default_id


@bp.route("/import", methods=["POST"])
@cross_origin(supports_credentials=True)
def import_ecosystem():

    # Extract arguments for the ecosystem to import
    # Either be a file location, YAML-as-string, or book_id and list of question uids

    yaml_string = request.files["file"].read()
    if "file" in request.files:
        (
            df_domain_,
            df_innovation_,
            df_questions_,
        ) = ecosystem_importer.parse_yaml_string(yaml_string)

    elif request.json is not None:
        yaml_filename = request.json.get("filename", None)
        yaml_string = request.json.get("yaml_string", None)
        book_id = request.json.get("book_id", None)
        exercise_list = request.json.get("question_list", None)

        if yaml_filename:
            (
                df_domain_,
                df_innovation_,
                df_questions_,
            ) = ecosystem_importer.parse_yaml_file(yaml_filename)
        elif yaml_string:
            (
                df_domain_,
                df_innovation_,
                df_questions_,
            ) = ecosystem_importer.parse_yaml_string(yaml_string)
        elif book_id and exercise_list:
            (
                df_domain_,
                df_innovation_,
                df_questions_,
            ) = ecosystem_importer.parse_content(book_id, exercise_list)

        else:
            return jsonify(
                {
                    "msg": (
                        "Could not process input. Provide either"
                        " a location of a YAML file,"
                        " a string of YAML content,"
                        " or a book_id and question_list"
                    )
                }
            )

    update_fixed_data(df_domain_, df_innovation_, df_questions_)

    return jsonify({"msg": "Ecosystem successfully imported"})


@bp.route("/datasets/feature_weights", methods=["POST"])
@cross_origin(supports_credentials=True)
def new_feature_weights_set():
    feature_weights_keys = set(current_app.config["DEFAULT_FEATURE_WEIGHTS"].keys())
    if not request.is_json:
        raise InvalidUsage(
            "Unable to load feature weights as json file.", status_code=404
        )
    else:
        new_feature_weights = request.json
        if set(new_feature_weights.keys()) != feature_weights_keys:
            raise InvalidUsage(
                "Incomplete or incorrect feature weight keys", status_code=400
            )
    feature_weight_id = store_feature_weights(new_feature_weights)
    return jsonify(
        {
            "msg": "Feature weights successfully imported.",
            "feature_weight_set_id": feature_weight_id,
        }
    )


@bp.route("/datasets/feature_weights/default", methods=["PUT"])
@cross_origin(supports_credentials=True)
def set_default_feature_weights_id():
    df = current_app.df
    if not request.is_json:
        raise InvalidUsage(
            "Unable to load new default id as json file.", status_code=404
        )
    else:
        new_default_id = request.json
        if new_default_id not in df["feature_weights"].keys():
            raise InvalidUsage("Feature weight id not found.", status_code=400)
    default_id = write_default_feature_weights_id(new_default_id)
    return jsonify(
        {
            "msg": "Successfully set default feature weight id.",
            "feature_weight_set_id": default_id,
        }
    )
