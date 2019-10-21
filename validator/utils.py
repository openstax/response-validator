# utils.py
# Author: drew
# Load up the relevant book and question data and transform into the
# simplified data frames we need for garbage detection

import pandas as pd
import re
import os
import string

# from nltk.corpus import words


def make_tristate(var, default=True):
    if type(default) == int:
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
    df_domain.replace(set(), "").to_csv(
        os.path.join(data_dir, "df_domain.csv"), index=None
    )
    df_innovation.replace(set(), "").to_csv(
        os.path.join(data_dir, "df_innovation.csv"), index=None
    )
    df_questions.replace(set(), "").to_csv(
        os.path.join(data_dir, "df_questions.csv"), index=None
    )


def get_fixed_data(data_dir):
    data_files = os.listdir(data_dir)
    files_to_find = ["df_innovation.csv", "df_domain.csv", "df_questions.csv"]
    num_missing_files = len(set(files_to_find) - set(data_files))
    if num_missing_files == 0:
        print(f"Loading existing data from {data_dir}...")
        df_innovation = pd.read_csv(os.path.join(data_dir, files_to_find[0]))
        df_domain = pd.read_csv(os.path.join(data_dir, files_to_find[1]))
        df_questions = pd.read_csv(os.path.join(data_dir, files_to_find[2]))
        # BBB Determine if these are "old" csv files, then rename columns and
        # other post-processing steps

        needs_rewrite = False
        if "CNX Book Name" in df_domain.columns:
            needs_rewrite = True
            # Get map of book names to newest vuid for that book
            book_ids_by_name = dict(
                sorted(
                    set(
                        df_innovation[df_innovation["module_id"].str.contains(":")][
                            ["subject_name", "module_id"]
                        ]
                        .apply(
                            lambda x: (x.subject_name, x.module_id.split(":")[0]),
                            axis=1,
                        )
                        .tolist()
                    ),
                    key=(lambda x: x[1].split("@")[1].split(".")[0]),
                )
            )
            # Fix up innovation dataframe:
            df_innovation = df_innovation[
                ["subject_name", "module_id", "innovation_words"]
            ].rename(columns={"subject_name": "book_name", "module_id": "cvuid"})

            # Use map to fix up domain words df
            df_domain = df_domain.apply(
                lambda x: (book_ids_by_name[x["CNX Book Name"]], x[0], x[1]),
                axis="columns",
                result_type="expand",
            ).rename(columns={0: "vuid", 1: "book_name", 2: "domain_words"})

            # Fix up questions - this is all done at ecosystem import, now

            df_questions["qid"] = df_questions["uid"].apply(lambda x: x.split("@")[0])
            df_questions["stem_words"] = (
                df_questions["stem_text"]
                .fillna("")
                .apply(lambda x: str(set(x.lower().translate(translator).split())))
            )
            df_questions["mc_words"] = (
                df_questions["option_text"]
                .fillna("")
                .apply(lambda x: str(set(x.lower().translate(translator).split())))
            )
            df_questions["contains_number"] = df_questions.apply(
                lambda x: contains_number(x), axis=1
            )

            df_questions = df_questions[
                [
                    "qid",
                    "uid",
                    "module_id",
                    "contains_number",
                    "mc_words",
                    "option_text",
                    "stem_words",
                    "stem_text",
                ]
            ].rename(columns={"module_id": "cvuid"})

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
        # Question qids are imported as ints - let's convert that to strings for comparison
        df_questions["qid"] = df_questions["qid"].apply(str)

        if needs_rewrite:
            print("old CVS files detected:")
            write_fixed_data(df_domain, df_innovation, df_questions, data_dir)

    else:
        print(f"No data loaded from {data_dir}: rolling with empty datasets")
        df_innovation = pd.DataFrame(columns=["cvuid", "innovation_words", "book_name"])
        df_domain = pd.DataFrame(columns=["vuid", "domain_words", "book_name"])
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

    return df_innovation, df_domain, df_questions
