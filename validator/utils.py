# utils.py
# Author: drew
# Load up the relevant book and question data and transform into the
# simplified data frames we need for garbage detection

import pandas as pd
import re
import os

# from nltk.corpus import words

import pkg_resources


def get_fixed_data():

    # Check to see if the dataframes already exist.  If so, load from disk.
    data_dir = pkg_resources.resource_filename("validator", "ml/data/")
    data_files = os.listdir(data_dir)
    files_to_find = ["df_innovation.csv", "df_domain.csv", "df_questions.csv"]
    num_missing_files = len(set(files_to_find) - set(data_files))
    if num_missing_files == 0:
        print("Loading existing data...")
        df_innovation = pd.read_csv(data_dir + files_to_find[0])
        df_domain = pd.read_csv(data_dir + files_to_find[1])
        df_questions = pd.read_csv(data_dir + files_to_find[2])
    else:
        print(
            "Can't find fixed data so creating from scratch . . . this may take a bit!"
        )
        df_innovation, df_domain, df_questions = create_fixed_data()
        df_innovation.to_csv(data_dir + files_to_find[0], index=None)
        df_domain.to_csv(data_dir + files_to_find[1], index=None)
        df_questions.to_csv(data_dir + files_to_find[2], index=None)
        print("Finished")

    df_questions["qid"] = df_questions["uid"].apply(lambda x: x.split("@")[0])

    return df_innovation, df_domain, df_questions


def create_fixed_data():

    data_dir = pkg_resources.resource_filename("validator", "ml/data/")
    corp_dir = pkg_resources.resource_filename("validator", "ml/corpora/")
    df_grouped = pd.read_csv(data_dir + "book_dataframe.csv")
    df_questions = pd.read_csv(data_dir + "df_questions.csv")
    common_vocab = pd.read_csv(corp_dir + "common.txt", header=None)[0].values.tolist()

    # Get domain-level and module-level vocabulary innovation
    # Computes words that are novel at that particular level (over general corpora)
    df_grouped["innovation_words"] = ""

    books = df_grouped["CNX Book Name"].unique()
    df_grouped_innovation = pd.DataFrame()
    df_domain = pd.DataFrame()

    for book in books:
        df_temp = df_grouped[df_grouped["CNX Book Name"] == book]
        frame_length = df_temp.shape[0]
        cumulative_word_set = set(common_vocab)
        for ll in range(0, frame_length):
            text = df_temp.iloc[ll].text.lower()
            text = re.sub('[!?().,;"“”:0-9]', " ", text)
            current_words = set(text.split())
            innovation_words = current_words - cumulative_word_set
            df_temp["innovation_words"].iloc[ll] = innovation_words
            cumulative_word_set = cumulative_word_set | current_words
        df_grouped_innovation = df_grouped_innovation.append(df_temp)
        df_domain = df_domain.append(
            pd.DataFrame(
                {
                    "CNX Book Name": book,
                    "domain_words": set.union(
                        *df_temp.innovation_words.values.tolist()
                    ),
                }
            ).iloc[0:1]
        )

    # Final stuff
    df_innovation = df_grouped_innovation.rename(
        columns={
            "CNX Book Name": "subject_name",
            "CNX Chapter Number": "chapter_id",
            "CNX Section Number": "section_id",
        }
    )
    df_innovation = df_innovation.drop("text", axis=1)

    return df_innovation, df_domain, df_questions
