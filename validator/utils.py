# utils.py
# Author: drew
# Load up the relevant book and question data and transform into the
# simplified data frames we need for garbage detection

import json
import re
import os
import string


import pandas as pd
from collections import OrderedDict


def make_tristate(var, default=True):
    if type(default) in (int, float):
        try:
            return int(var)
        except ValueError:
            pass
        try:
            return float(var)
        except:  # noqa
            pass
    if var == "auto" or type(var) == bool:
        return var
    elif var in ("False", "false", "f", "0", "None", ""):
        return False
    elif var in ("True", "true", "t", "1"):
        return True
    else:
        return default


def contains_number(df_row):
    math_words = [
        "meter",
        "newton",
        "time",
        "rate",
        "variable",
        "unit",
        "contant",
        "meter",
        "charge",
    ]
    qtext = " ".join((df_row.stem_text, df_row.option_text)).lower()
    if "contains_number" in df_row:
        return df_row["contains_number"]
    elif re.search(r"[\+\-\*\=\/\d]", qtext) is not None:
        return True
    else:
        return any([m in qtext for m in math_words])


translator = str.maketrans("", "", string.punctuation)


def split_to_words(df, text_column):
    return (
        df[text_column]
        .fillna("")
        .apply(lambda x: x.lower().translate(translator).split())
    )


def write_fixed_data(df_domain, df_innovation, df_questions, data_dir):
    print(f"Writing data to: {data_dir}")
    if df_domain is not None:
        df_domain.replace(set(), "").to_csv(
            os.path.join(data_dir, "df_domain.csv"), index=None
        )
    if df_innovation is not None:
        df_innovation.replace(set(), "").to_csv(
            os.path.join(data_dir, "df_innovation.csv"), index=None
        )
    if df_questions is not None:
        df_questions.replace(set(), "").to_csv(
            os.path.join(data_dir, "df_questions.csv"), index=None
        )


def write_feature_weights(feature_weights, data_dir):
    print(f"Writing data to: {data_dir}")
    with open(os.path.join(data_dir, "feature_weights.json"), "w") as f:
        json.dump(feature_weights, f, indent=2)


def get_fixed_data(data_dir):
    data_files = os.listdir(data_dir)
    files_to_find = [
        "df_innovation.csv",
        "df_domain.csv",
        "df_questions.csv",
    ]
    missing_files = set(files_to_find) - set(data_files)
    num_missing_files = len(missing_files)
    if num_missing_files == 0:
        print(f"Loading existing data from {data_dir}...")
        df_innovation = pd.read_csv(os.path.join(data_dir, files_to_find[0]))
        df_domain = pd.read_csv(os.path.join(data_dir, files_to_find[1]))
        df_questions = pd.read_csv(os.path.join(data_dir, files_to_find[2]))

        # Convert domain and innovation words from comma-separated strings to set
        # This works in memory just fine but won't persist in file
        df_domain = df_domain.fillna("")
        df_innovation = df_innovation.fillna("")
        df_questions = df_questions.fillna("")

        # Tests for feature weight id column and adds it to domain if it's not in there yet
        if "feature_weights_id" not in df_domain.columns:
            df_domain["feature_weights_id"] = None

        df_domain["domain_words"] = df_domain["domain_words"].apply(
            lambda x: set([w[1:-1] for w in x[1:-1].split(", ")])
        )
        df_innovation["innovation_words"] = df_innovation["innovation_words"].apply(
            lambda x: set([w[1:-1] for w in x[1:-1].split(", ")])
        )

        df_questions["stem_words"] = df_questions["stem_words"].apply(
            lambda x: set([w[1:-1] for w in x[1:-1].split(", ")])
        )

        df_questions["mc_words"] = df_questions["mc_words"].apply(
            lambda x: set([w[1:-1] for w in x[1:-1].split(", ")])
        )
        # Question qids are imported as ints - let's convert that to strings for comparison
        df_questions["qid"] = df_questions["qid"].apply(str)

    else:
        print(
            f"""No vocab data loaded from {data_dir}:  missing {', '.join(missing_files)}"""
            """\nrolling with empty datasets"""
        )
        df_innovation = pd.DataFrame(columns=["cvuid", "innovation_words", "book_name"])
        df_domain = pd.DataFrame(
            columns=["vuid", "domain_words", "book_name", "feature_weights_id"]
        )
        df_questions = pd.DataFrame(
            columns=[
                "contains_number",
                "cvuid",
                "mc_words",
                "option_text",
                "qid",
                "stem_text",
                "stem_words",
                "uid",
            ]
        )

    # Now load the feature_weights and default feature_weight_id, if found.
    try:
        with open(os.path.join(data_dir, "feature_weights.json")) as f:
            feature_weights = json.load(f, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        print("No feature weights loaded, using defaults")
        feature_weights = OrderedDict()

    return df_innovation, df_domain, df_questions, feature_weights
