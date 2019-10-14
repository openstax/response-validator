import pandas as pd
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

df = {}
df["innovation"] = pd.DataFrame(columns=["cvuid", "innovation_words", "book_name"])
df["domain"] = pd.DataFrame(columns=["vuid", "domain_words", "book_name"])
df["questions"] = pd.DataFrame(columns=["contains_number", "cvuid", "mc_words", "option_text",
                                        "qid", "stem_text", "stem_words", "uid"])
