# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs
import os
import sys
import sentry_sdk

from sentry_sdk.integrations.flask import FlaskIntegration
from flask import Flask

from .utils import get_fixed_data

from . import read_api, write_api, validate_api, training_api


def create_app(**kwargs):
    app = Flask(__name__.split(".")[0])
    app.url_map.strict_slashes = False
    app.config.from_pyfile("validator.cfg")
    app.config.from_object("validator.default_settings")
    app.config.from_envvar("VALIDATOR_SETTINGS", silent=True)
    app.config.from_envvar("VALIDATOR_CONFIG", silent=True)

    if kwargs:
        app.config.from_mapping(kwargs)

    # Get the global data for the app:
    #    innovation words by page,
    #    domain words by subject/book,
    #    and table linking question uid to page-in-book id
    data_dir = app.config.get("DATA_DIR", "")

    try:
        os.listdir(data_dir)
    except FileNotFoundError:
        raise FileNotFoundError("Bad or no DATA_DIR defined")

    df_innovation_, df_domain_, df_questions_, feature_weights = get_fixed_data(
        data_dir
    )
    if "default_id" not in feature_weights:
        feature_weights_key = app.config.get("DEFAULT_FEATURE_WEIGHTS_KEY")
        if feature_weights_key not in feature_weights:
            feature_weights[feature_weights_key] = app.config.get(
                "DEFAULT_FEATURE_WEIGHTS"
            )
            feature_weights["default_id"] = feature_weights_key

    datasets = {}
    datasets["innovation"] = df_innovation_
    datasets["domain"] = df_domain_
    datasets["questions"] = df_questions_
    datasets["feature_weights"] = feature_weights
    app.datasets = datasets

    app.qids = {}
    for idcol in ("uid", "qid"):
        app.qids[idcol] = set(datasets["questions"][idcol].values.tolist())

    app.register_blueprint(read_api.bp)
    app.register_blueprint(write_api.bp)
    app.register_blueprint(validate_api.bp)
    app.register_blueprint(training_api.bp)

    sentry_sdk.init(
      dsn=current_app.config["SENTRY_DSN"],
      integrations=[FlaskIntegration()]
    )

    return app


if __name__ == "__main__":
    app = create_app(**dict([a.split("=") for a in sys.argv[1:]]))
    app.run()
