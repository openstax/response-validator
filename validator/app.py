# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import Flask

from .utils import get_fixed_data

import pkg_resources

from . import df, read_api, write_api, validate_api, training_api


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object("validator.default_settings")
    app.config.from_envvar("VALIDATOR_SETTINGS", silent=True)
    if test_config is not None:
        app.config.from_mapping(test_config)

    # Get the global data for the app:
    #    innovation words by page,
    #    domain words by subject/book,
    #    and table linking question uid to page-in-book id
    data_dir = app.config.get(
        "DATA_DIR", pkg_resources.resource_filename("validator", "ml/data/")
    )

    df_innovation_, df_domain_, df_questions_ = get_fixed_data(data_dir)

    df["innovation"] = df_innovation_
    df["domain"] = df_domain_
    df["questions"] = df_questions_

    app.register_blueprint(read_api.bp)
    app.register_blueprint(write_api.bp)
    app.register_blueprint(validate_api.bp)
    app.register_blueprint(training_api.bp)

    return app
