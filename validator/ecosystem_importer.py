import pandas as pd
import numpy as np
import requests
import re
import yaml

from .utils import split_to_words, contains_number


class EcosystemImporter(object):
    def __init__(
        self,
        base_exercise_url="https://exercises.openstax.org/api/exercises?q=uid:%22{}%22",
        common_vocabulary_filename=None,
        common_vocabulary_list=[],
    ):

        self.base_exercise_url = base_exercise_url

        if common_vocabulary_filename:
            with open(common_vocabulary_filename, "r") as f:
                words = f.read()
            words = self.get_words(words)
            self.common_vocabulary_set = set(words)
        else:
            self.common_vocabulary_set = set(common_vocabulary_list)

    def get_words(self, text_str):
        return re.findall("[a-z]+", text_str.lower())

    def flatten_to_leaves(self, node):
        if "contents" in node:
            leaves = []
            for child in node["contents"]:
                leaves.extend(self.flatten_to_leaves(child))
            return leaves
        else:
            return [node]

    def format_cnxml(self, text):
        clean = re.compile("<.*?>")
        clean_text = re.sub(clean, " ", text)
        clean_text = clean_text.replace("\n", " ")
        clean_text = clean_text.replace("\\text{", " ")
        clean_text = clean_text.replace("}", " ")
        return clean_text

    def get_page_content(self, book_cnx_id, page_id, archive_url):
        full_id = "{}:{}".format(book_cnx_id, page_id)
        content = requests.get(archive_url.format(full_id)).json()["content"]
        return self.format_cnxml(content)

    def diff_book_dataframe(self, book_dataframe):
        # Iterate through the pages in the book dataframe
        # Get innovation words for each page (removing previous words + common vocab)
        current_vocab = self.common_vocabulary_set
        innovation_words = []
        for ii in range(0, book_dataframe.shape[0]):
            page_words = self.get_words(book_dataframe.iloc[ii]["content"])
            page_words = set(page_words)
            new_words = page_words - current_vocab
            innovation_words.append(new_words)
            current_vocab = current_vocab | new_words
        book_dataframe["innovation_words"] = innovation_words
        return book_dataframe

    def get_book_content(self, archive_url, book_cnx_id):
        # Get the tree object from the book_cnx_id
        # Flatten this out to a list of linearly arranged page ids
        # Then grab all of the content for each id, weave into a pandas dataframe
        resp = requests.get(archive_url.format(book_cnx_id))
        node = resp.json()["tree"]
        node_list = self.flatten_to_leaves(node)
        id_list = [n["id"] for n in node_list]
        content = [
            self.get_page_content(book_cnx_id, page_id, archive_url)
            for page_id in id_list
        ]
        book_dataframe = pd.DataFrame(
            {
                "book_id": [book_cnx_id] * len(id_list),
                "page_id": id_list,
                "content": content,
            }
        )

        df_innovation = self.diff_book_dataframe(book_dataframe)
        df_innovation["cvuid"] = df_innovation.apply(
            lambda x: x.book_id + ":" + x.page_id, axis=1
        )
        df_innovation = df_innovation[["cvuid", "innovation_words"]]
        all_vocab = set(df_innovation["innovation_words"].apply(list).agg("sum"))
        df_domain = pd.DataFrame({"vuid": [book_cnx_id], "domain_words": [all_vocab]})

        return df_domain, df_innovation

    def get_question_content(self, question_uid_list, book_id, module_id_set):
        # Each uid may consist of multiple "questions"
        # For each question, grab the stem_html
        # Also, concatenate all the content_html in "answers"
        N_chunk = (
            100
        )  # Limit of the API server on how many exercises we can get at a time
        question_list_chunks = [
            question_uid_list[x : x + N_chunk]
            for x in range(0, len(question_uid_list), N_chunk)
        ]
        item_list = []
        for sublist in question_list_chunks:
            question_list_str = ",".join(sublist)
            question_json = requests.get(
                self.base_exercise_url.format(question_list_str)
            )
            item_list.extend(question_json.json()["items"])

        # Now iterate through all items and questions within items
        # For each item/question pair extract the clean stem_html,
        #  and cleaned (joined) answers
        uid_list = []
        stem_list = []
        answer_list = []
        module_id_list = []
        for item in item_list:
            uid = item["uid"]
            for question in item["questions"]:
                stem_text = self.format_cnxml(question["stem_html"])
                answer_text = " ".join(
                    [
                        self.format_cnxml(answer["content_html"])
                        for answer in question["answers"]
                    ]
                )
                modules_in_tags = [t for t in item["tags"] if "context-cnxmod" in t]
                modules_in_tags = set([t.split(":")[1] for t in modules_in_tags])
                target_module_id = modules_in_tags & module_id_set
                if len(target_module_id) == 0:
                    target_module_id = np.nan
                else:
                    target_module_id = list(target_module_id)[0]
                module_id = target_module_id
                uid_list.append(uid)
                stem_list.append(stem_text)
                answer_list.append(answer_text)
                module_id_list.append(module_id)
        question_df = pd.DataFrame(
            {
                "uid": uid_list,
                "module_id": module_id_list,
                "stem_text": stem_list,
                "option_text": answer_list,
            }
        )
        question_df["qid"] = question_df["uid"].apply(lambda x: x.split("@")[0])
        question_df["stem_words"] = split_to_words(question_df, "stem_text")
        question_df["mc_words"] = split_to_words(question_df, "option_text")
        question_df["contains_number"] = question_df.apply(
            lambda x: contains_number(x), axis=1
        )

        return question_df

    def parse_content(
        self,
        book_id,
        question_uid_list,
        book_title,
        archive_url="https://archive.cnx.org",
    ):

        df_domain, df_innovation = self.get_book_content(archive_url, book_id)
        df_domain["book_name"] = book_title
        df_innovation["book_name"] = book_title

        module_id_set = (
            df_innovation["cvuid"].apply(lambda x: x.split(":")[1]).values.tolist()
        )
        unversioned_module_id_set = [m.split("@")[0] for m in module_id_set]
        module_id_df = pd.DataFrame(
            {"vers_module_id": module_id_set, "module_id": unversioned_module_id_set}
        )
        df_questions = self.get_question_content(
            question_uid_list, book_id, set(unversioned_module_id_set)
        )
        df_questions = df_questions.merge(module_id_df)
        df_questions["cvuid"] = df_questions.apply(
            lambda x: book_id + ":" + x.vers_module_id, axis=1
        )
        df_questions = df_questions[
            [
                "qid",
                "contains_number",
                "uid",
                "cvuid",
                "mc_words",
                "stem_words",
                "stem_text",
                "option_text",
            ]
        ]

        return df_domain, df_innovation, df_questions

    def parse_yaml_content(self, yaml_content):

        book_title = yaml_content["title"]
        archive_url = yaml_content["books"][0]["archive_url"] + "/contents/{}"
        book_cnx_id = yaml_content["books"][0]["cnx_id"]
        question_uid_list = yaml_content["books"][0]["exercise_ids"]

        # Strip ' (uuid@ver)' from end of title in yaml: 'book name (uuid@ver)'
        if book_cnx_id in book_title:
            book_title = book_title[: book_title.find(book_cnx_id) - 2]

        return self.parse_content(
            book_cnx_id, question_uid_list, book_title, archive_url
        )

    def parse_yaml_string(self, yaml_string):

        data_loaded = yaml.safe_load(yaml_string)

        return self.parse_yaml_content(data_loaded)

    def parse_yaml_file(self, yaml_filename):

        # Use the yaml library to parse the file into a dictionary
        with open(yaml_filename, "r") as stream:
            data_loaded = yaml.safe_load(stream)
            return self.parse_yaml_content(data_loaded)
