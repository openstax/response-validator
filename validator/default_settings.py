import collections

DATA_DIR = "validator/ml/data"

PARSER_DEFAULTS = {
    "remove_stopwords": True,
    "tag_numeric": "auto",
    "spelling_correction": "auto",
    "remove_nonwords": True,
    "spell_correction_max": 10,
    "lazy_math_mode": True,
}

SPELLING_CORRECTION_DEFAULTS = {
    "spell_correction_max_edit_distance": 3,
    "spell_correction_min_word_length": 5,
}

# If number, feature is used and has the corresponding weight.
# A value of 0 indicates that the feature won't be computed
VALIDITY_FEATURE_DICT = collections.OrderedDict(
    {
        "stem_word_count": 0,
        "option_word_count": 0,
        "innovation_word_count": 2.2,
        "domain_word_count": 2.5,
        "bad_word_count": -3,
        "common_word_count": 0.7,
        "intercept": 0,
    }
)
