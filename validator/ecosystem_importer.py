import pandas as pd
import requests
import re
import yaml

# MAKE INTO A CLASS


class EcosystemImporter(object):
    def __init__(
        self,
        base_exercise_url="https://exercises.openstax.org/api/exercises?q=uid:%22{}%22",
        common_vocabulary_filename=None,
        common_vocabulary_list=[],
    ):

        self.base_exercise_url = base_exercise_url

        if common_vocabulary_filename:
            f = open(common_vocabulary_filename, "r")
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
            innovation_words.append(",".join(list(new_words)))
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
        all_vocab = (
            df_innovation.groupby("cvuid")["innovation_words"].agg("sum").values[0]
        )
        df_domain = pd.DataFrame({"book_id": [book_cnx_id], "domain_words": all_vocab})

        return df_domain, df_innovation

    def get_question_content(self, question_uid_list):
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
        # For each item/question pair extract the clean stem_html, and cleaned (joined) answers
        uid_list = []
        stem_list = []
        answer_list = []
        for item in item_list:
            uid = item["uid"]
            L = len(item["questions"])
            for question in item["questions"]:
                stem_text = self.format_cnxml(question["stem_html"])
                answer_text = " ".join(
                    [
                        self.format_cnxml(answer["content_html"])
                        for answer in question["answers"]
                    ]
                )
                uid_list.append(uid)
                stem_list.append(stem_text)
                answer_list.append(answer_text)
        question_df = pd.DataFrame(
            {"uid": uid_list, "stem_text": stem_list, "answer_text": answer_list}
        )

        return question_df

    def parse_content(
        self, book_id, question_uid_list, archive_url="https://archive.cnx.org"
    ):

        df_domain, df_innovation = self.get_book_content(archive_url, book_id)
        df_questions = self.get_question_content(question_uid_list)

        return df_domain, df_innovation, df_questions

    def parse_yaml(self, yaml_filename):

        # Use the yaml library to parse the file into a dictionary
        with open(yaml_filename, "r") as stream:
            data_loaded = yaml.safe_load(stream)

            book_title = data_loaded["title"]
            archive_url = data_loaded["books"][0]["archive_url"] + "/contents/{}"
            book_cnx_id = data_loaded["books"][0]["cnx_id"]
            question_uid_list = data_loaded["books"][0]["exercise_ids"]

            return self.parse_content(book_cnx_id, question_uid_list, archive_url)
