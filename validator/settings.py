from collections import OrderedDict
from dotenv import load_dotenv
from os import environ, getenv

load_dotenv()

def getenvbool(key, default):
    return str(getenv(key, default)).lower() == "true"

if "DATA_DIR" in environ:
    DATA_DIR = getenv("DATA_DIR")
elif "DATA_BUCKET_NAME" in environ:
    DATA_DIR = f's3://{getenv("DATA_BUCKET_NAME")}/{getenv("ENVIRONMENT_NAME", "development")}'
else:
    DATA_DIR = "validator/ml/data"

PARSER_DEFAULTS = {
    "remove_stopwords": getenvbool("PARSER_DEFAULTS_REMOVE_STOPWORDS", True),
    "tag_numeric": getenv("PARSER_DEFAULTS_TAG_NUMERIC", "auto"),
    "spelling_correction": getenv("PARSER_DEFAULTS_SPELLING_CORRECTION", "auto"),
    "remove_nonwords": getenvbool("PARSER_DEFAULTS_REMOVE_NONWORDS", True),
    "spell_correction_max": int(getenv("PARSER_DEFAULTS_SPELL_CORRECTION_MAX", 10)),
    "lazy_math_mode": getenvbool("PARSER_DEFAULTS_LAZY_MATH_MODE", True),
}

SPELLING_CORRECTION_DEFAULTS = {
    "spell_correction_max_edit_distance": int(getenv(
        "SPELLING_CORRECTION_DEFAULTS_SPELL_CORRECTION_MAX_EDIT_DISTANCE", 3)),
    "spell_correction_min_word_length": int(getenv(
        "SPELLING_CORRECTION_DEFAULTS_SPELL_CORRECTION_MIN_WORD_LENGTH", 5)),
}

# If number, feature is used and has the corresponding weight.
# A value of 0 indicates that the feature won"t be computed
DEFAULT_FEATURE_WEIGHTS = OrderedDict(
    {
        "stem_word_count": float(getenv("DEFAULT_FEATURE_WEIGHTS_STEM_WORD_COUNT", 0)),
        "option_word_count": float(getenv("DEFAULT_FEATURE_WEIGHTS_OPTION_WORD_COUNT", 0)),
        "innovation_word_count": float(getenv(
            "DEFAULT_FEATURE_WEIGHTS_INNOVATION_WORD_COUNT", 2.2)),
        "domain_word_count": float(getenv("DEFAULT_FEATURE_WEIGHTS_DOMAIN_WORD_COUNT", 2.5)),
        "bad_word_count": float(getenv("DEFAULT_FEATURE_WEIGHTS_BAD_WORD_COUNT", -3)),
        "common_word_count": float(getenv("DEFAULT_FEATURE_WEIGHTS_COMMON_WORD_COUNT", 0.7)),
    }
)

DEFAULT_FEATURE_WEIGHTS_KEY = getenv(
    "DEFAULT_FEATURE_WEIGHTS_KEY", "d3732be6-a759-43aa-9e1a-3e9bd94f8b6b")
