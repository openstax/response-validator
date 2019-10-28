# unsupervised_garbage_detection.py
# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garbage detection variants and simulates
# accuracy/complexity tradeoffs

from flask import jsonify, request, Blueprint, current_app
from flask_cors import cross_origin


import pkg_resources

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


from .validate_api import validate_response
from .utils import make_tristate


bp = Blueprint("training_api", __name__, url_prefix="/")


CORPORA_PATH = pkg_resources.resource_filename("validator", "ml/corpora")


@bp.route("/train", methods=("GET", "POST"))
@cross_origin(supports_credentials=True)
def validation_train():

    # Read out the parser and classifier settings from the path arguments
    if request.method == "POST":
        args = request.form
    else:
        args = request.args
    train_feature_dict = {
        key: make_tristate(args.get(key, val), val)
        for key, val in current_app.config["VALIDITY_FEATURE_DICT"].items()
    }
    features_to_consider = [
        k for k in train_feature_dict.keys() if train_feature_dict[k]
    ]
    if ("intercept") in features_to_consider:
        features_to_consider.remove("intercept")
    parser_params = {
        key: make_tristate(args.get(key, val), val)
        for key, val in current_app.config["PARSER_DEFAULTS"].items()
    }
    cv_input = args.get("cv", 5)

    # Read in the dataframe of responses from json input
    response_df = request.json.get("response_df", None)
    response_df = pd.read_json(response_df).reset_index()

    # Parse the responses in response_df to get counts on the various word categories
    # Map the valid label of the input to the output
    output_df = response_df.apply(
        lambda x: validate_response(
            x.free_response, x.uid, train_feature_dict, **parser_params
        ),
        axis=1,
    )
    output_df = pd.DataFrame(list(output_df))
    output_df["valid_label"] = response_df["valid_label"]

    # Do an N-fold cross validation if cv > 1.
    # Then get coefficients/intercept for the entire dataset
    lr = LogisticRegression(
        solver="saga", max_iter=1000, fit_intercept=train_feature_dict["intercept"] != 0
    )
    X = output_df[features_to_consider].values
    y = output_df["valid_label"].values

    validation_array = -1
    if cv_input > 1:
        validation_array = cross_val_score(lr, X, y, cv=cv_input)
    lr.fit(X, y)
    coef = lr.coef_
    intercept = lr.intercept_[0]
    validation_score = float(np.mean(validation_array))

    # Create the return dictionary with the coefficients/intercepts as well as
    # the parsed datafrane We really don't need to the return the dataframe but
    # it's nice for debugging!

    return_dictionary = dict(zip(features_to_consider, coef[0].tolist()))
    return_dictionary["intercept"] = intercept
    return_dictionary["output_df"] = output_df.to_json()
    return_dictionary["cross_val_score"] = validation_score
    return jsonify(return_dictionary)
