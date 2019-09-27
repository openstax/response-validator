# utils.py
# Author: drew
# Load up the relevant book and question data and transform into the
# simplified data frames we need for garbage detection

import pandas as pd
import re
import os
import string

# from nltk.corpus import words

import pkg_resources


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
    if df_row["contains_number"]:
        return True
    elif re.search(r"[\+\-\*\=\/\d]", df_row.text) is not None:
        return True
    else:
        text = df_row.text.lower()
        return any([m in text for m in math_words])


def write_fixed_data(df_domain, df_innovation, df_questions):
    data_dir = pkg_resources.resource_filename("validator", "ml/data/")
    df_domain.to_csv(data_dir + 'df_domain.csv', index=None)
    df_innovation.to_csv(data_dir + 'df_innovation.csv', index=None)
    df_questions.to_csv(data_dir + 'df_questions.csv', index=None)


def get_fixed_data():
    data_dir = pkg_resources.resource_filename("validator", "ml/data/")
    data_files = os.listdir(data_dir)
    files_to_find = ["df_innovation.csv", "df_domain.csv", "df_questions.csv"]
    num_missing_files = len(set(files_to_find) - set(data_files))
    if num_missing_files == 0:
        print("Loading existing data...")
        df_innovation = pd.read_csv(data_dir + files_to_find[0])
        df_domain = pd.read_csv(data_dir + files_to_find[1])
        df_questions = pd.read_csv(data_dir + files_to_find[2])
        # Convert domain and innovation words from comma-separated strings to list
        # This works in memory just fine but won't persist in file
        df_domain = df_domain.fillna('')
        df_innovation = df_innovation.fillna('')

        df_domain['domain_words'] = df_domain['domain_words'].apply(
            lambda x: x.split(',')
        )
        df_innovation['innovation_words'] = df_innovation['innovation_words'].apply(
            lambda x: x.split(',')
        )

        df_questions["qid"] = df_questions["uid"].apply(
            lambda x: x.split("@")[0]
        )
        translator = str.maketrans("", "", string.punctuation)
        df_questions["stem_words"] = df_questions["stem_text"].fillna('').apply(
            lambda x: set(x.lower().translate(translator).split(','))
        )
        df_questions["mc_words"] = df_questions["option_text"].fillna('').apply(
            lambda x: set(x.lower().translate(translator).split(','))
        )
        df_questions["contains_number"] = df_questions.apply(
            lambda x: contains_number(x), axis=1
        )
    else:
        print(
            "Can't find fixed data so creating from scratch . . . this may take a bit!"
        )
        df_innovation = pd.DataFrame()
        df_domain = pd.DataFrame()
        df_questions = pd.DataFrame()

    return df_innovation, df_domain, df_questions
