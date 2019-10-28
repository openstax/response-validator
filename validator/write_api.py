# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import jsonify, request, Blueprint, current_app
from flask_cors import cross_origin

import pkg_resources

from .ecosystem_importer import EcosystemImporter
from .utils import write_fixed_data


CORPORA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")

bp = Blueprint("write_api", __name__, url_prefix="/")

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


@bp.route("/import", methods=["POST"])
@cross_origin(supports_credentials=True)
def import_ecosystem():

    # Extract arguments for the ecosystem to import
    # Either be a file location, YAML-as-string, or book_id and list of question uids

    yaml_string = request.files["file"].read()
    if "file" in request.files:
        df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_yaml_string(
            yaml_string
        )

    elif request.json is not None:
        yaml_filename = request.json.get("filename", None)
        yaml_string = request.json.get("yaml_string", None)
        book_id = request.json.get("book_id", None)
        exercise_list = request.json.get("question_list", None)

        if yaml_filename:
            df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_yaml_file(
                yaml_filename
            )
        elif yaml_string:
            df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_yaml_string(
                yaml_string
            )
        elif book_id and exercise_list:
            df_domain_, df_innovation_, df_questions_ = ecosystem_importer.parse_content(
                book_id, exercise_list
            )

        else:
            return jsonify(
                {
                    "msg": "Could not process input. Provide either"
                    " a location of a YAML file,"
                    " a string of YAML content,"
                    " or a book_id and question_list"
                }
            )

    update_fixed_data(df_domain_, df_innovation_, df_questions_)

    return jsonify({"msg": "Ecosystem successfully imported"})
