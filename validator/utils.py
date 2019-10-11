# utils.py
# Author: drew
# Load up the relevant book and question data and transform into the
# simplified data frames we need for garbage detection

import pandas as pd
import re
import os
import string

# from nltk.corpus import words


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
    elif (
        re.search(r"[\+\-\*\=\/\d]", qtext)
        is not None
    ):
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
    df_domain.replace(set(), "").to_csv(os.path.join(data_dir, "df_domain.csv"), index=None)
    df_innovation.replace(set(), "").to_csv(os.path.join(data_dir, "df_innovation.csv"), index=None)
    df_questions.replace(set(), "").to_csv(os.path.join(data_dir, "df_questions.csv"), index=None)


def get_fixed_data(data_dir):
    data_files = os.listdir(data_dir)
    files_to_find = ["df_innovation.csv", "df_domain.csv", "df_questions.csv"]
    num_missing_files = len(set(files_to_find) - set(data_files))
    if num_missing_files == 0:
        print("Loading existing data...")
        df_innovation = pd.read_csv(os.path.join(data_dir, files_to_find[0]))
        df_domain = pd.read_csv(os.path.join(data_dir, files_to_find[1]))
        df_questions = pd.read_csv(os.path.join(data_dir, files_to_find[2]))
        # Convert domain and innovation words from comma-separated strings to set
        # This works in memory just fine but won't persist in file
        df_domain = df_domain.fillna("")
        df_innovation = df_innovation.fillna("")
        df_questions = df_questions.fillna("")

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

    else:
        print("No data loaded: rolling with empty datasets")
        df_innovation = pd.DataFrame(columns=["cvuid", "innovation_words", "book_name"])
        df_domain = pd.DataFrame(columns=["vuid", "domain_words", "book_name"])
        df_questions = pd.DataFrame(columns=["contains_number", "cvuid", "mc_words", "option_text",
                                             "qid", "stem_text", "stem_words", "uid"])

    return df_innovation, df_domain, df_questions
