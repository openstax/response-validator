# response_validation_app

Implements a simple unsupervised method for classifying student short to medium sized responses to questions.

## Installation

This was developed in Python 3.6. 

It may be installed as a package from the pypi repository, using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install response-validator
```

## Development
After cloning the repository, you can install the repo in editable mode, as so:

```bash
pip install -e .
```
Note that this step will download several NLTK corpora, silently, and add them to the deployed tree.

Additional functionality for running algorith tests, etc. can be enabled by installing additional libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Development
Once installed, `python -m validator.app` will run the Flask dev webserver.

Note that this will launch a server with _no_ loaded book-specific data, and
will use a temporary directory, _that it will delete on exit_ to store any
uploaded/imported book data. In order to persist the book vocabulary data
between invocations, the `DATA_DIR` setting needs to be set to a path pointing
to an existing directory. This can be set in several ways.

1. set the VALIDATOR_SETTINGS environment variable to the path of a file
that contains the DATA_DIR setting:

`VALIDATOR_SETTINGS=data/dev.cfg python -m validator.app`

Where the contents of dev.cfg is:

```
DATA_DIR=data
```
and that directory `data` exists.

2. Pass a key-value command line argument:


`python -m validator.app DATA_DIR=data`


3. use gunicorn, provide arguments to app factory:


`gunicorn 'validator.app:create_app(DATA_DIR="data")'`


4. Use gunicorn, with an environment variable pointing to a config file:


`VALIDATOR_SETTINGS=../data/dev.cfg gunicorn "validator.app:create_app()"`

Note that this one can get confusing with relative paths, since flask  uses
directory the app is imported from (in this case, `validator`) as the config path when interpreting environment variables, while paths inside such files will be based on the python current working directory. When in doubt, use full paths:

`VALIDATOR_SETTINGS="$PWD/data/dev.cfg" gunicorn "validator.app:create_app()"`

### Production
The recommended production method for deployment is to use a WSGI compliant
server, such as gunicorn:

```bash
pip install gunicorn gevent
gunicorn -k gevent -b 5000 "validator.app:create_app(DATA_DIR='/var/lib/validator/data')" 
```

Ideally, use a socket, and place nginx or other webserver in front of flask, for https termination, if nothing else.

```bash
gunicorn -k gevent --bind /run/gunicorn.sock "validator.app:create_app(DATA_DIR='/var/lib/validator/data')"
```
## API

### Response Validation
The main route for the app is /validate, which accepts a plaintext response (`response`) that will be checked.  It can also accept a number of optional arguments:

- `uid` (e.g., '1000@1', default None): This is the uid for the question pertaining to the response. The uid is used to compute domain-specific and module-specific vocabulary to aid in the classification process.
Iff the version of the question specified is not available, any version of the same qid (question id without the version, e.g. 1000) will be used. 

- `remove_stopwords` (True or False, default True): Whether or not stopwords (e.g., 'the', 'and', etc) will be removed from the response.  This is generally advised since these words carry little predictive value.

- `tag_numeric` (True, False or auto, default auto): Whether numerical values will be tagged (e.g., 123.7 is tagged with a special 'numeric_type_float' identifier). While there are certainly responses for which this would be helpful, a large amount of student garbage consists of random number pressing which limits the utility of this option. Auto enables a mode that only does numeric tag processing if the question this response pertains to (as fond via the uid above) requires a numeric answer.

- `spelling_correction` (True, False or auto, default auto): Whether the app will attempt to correct misspellings. This is done by identifying unknown words in the response and seeing if a closely related known word can be substituted.  Currently, the app only attempts spelling correction on words of at least 5 characters in length and only considers candidate words that are within an edit distance of 2 from the misspelled word. When running in `auto` mode, the app will attempt to determine validity without spelling correction. Only if that is not valid, will it attempt to reassess validity with spelling correction.

- `spell_correction_max` (integer, default 10): Limit spelling corrections applied to this number.

- `remove_nonwords` (True or False, default True): Words that are not recognized (after possibly attempting spelling correction) are flagged with a special 'nonsense_word' tag.  This is done primarily to combat keyboard mashes (e.g., 'asdfljasdfk') that make a large percentage of invalid student responses.

Once the app is running, you can send requests using curl, requests, etc.  Here is an example using Python's requests library:

Here an example of how to call things using the Python requests library (assuming that the app is running on the default local development port):

```python
import json
import requests
params = {'response': 'This is my answar to the macromolecules question nitrogenous awerawfsfs'
          'uid': '100@2',
          'remove_stopwords': True,
          'tag_numeric=True': False,
          'spelling_correction': True,
          'remove_nonwords': True}
r = requests.get('http://127.0.0.1:5000/validate', params=params)
print(json.dumps(r.json(), indent=2))
{
  "bad_word_count": 1,
  "common_word_count": 3,
  "computation_time": 0.013212919235229492,
  "domain_word_count": 1,
  "inner_product": 1.5999999999999996,
  "innovation_word_count": 0,
  "intercept": 1,
  "lazy_math_evaluation": true,
  "num_spelling_correction": 2,
  "option_word_count": 0,
  "processed_response": "answer macromolecules question nitrogenous nonsense_word",
  "remove_nonwords": true,
  "remove_stopwords": true,
  "response": "This is my answar to the macromolecules question nitrogenous awerawfsfs",
  "spelling_correction": true,
  "spelling_correction_used": true,
  "stem_word_count": 0,
  "tag_numeric": "auto",
  "tag_numeric_input": "auto",
  "uid_found": true,
  "uid_used": "100@7",
  "valid": true,
  "version": "2.3.0"
}
```

As you can see from these results, a number of features are taken into account
when determining the potential validity of the students response: the words in
the response itself, the words from the associated question (stem words) and
its answers (option words), the words in the textbook associated with this
assignment (domain words), and the words in the textbook whose first appearance
is on the page associated with this question (innovation words). Various other
features (presence or absence of math, spelling correction, stop word
elimination, etc) are also applied. These tests depend on vocabularies being loaded
for each exercise.

## Service APIs

|Route|Response|Purpose|
|-----|--------|-------|
|`/ping`| `pong`| Determining that the validation service is operational.|
|`/version` or `/rev.txt`| version string (i.e. 2.3.0)|What version of service is installed|
|`/status`| json response (see below)| Detailed service info (extended version, start time) and datasets|


Here is the `/status` response for a server started on Oct 1, with a clean install of version 2.3.0,
and vocabularies for 5 books loaded:

```json
{
  "datasets": {
    "books": [
      {
        "name": "Biology 2e",
        "vocabularies": [
          "domain",
          "innovation"
        ],
        "vuid": "8d50a0af-948b-4204-a71d-4826cba765b8@15.45"
      },
      {
        "name": "College Physics for AP® Courses",
        "vocabularies": [
          "domain",
          "innovation"
        ],
        "vuid": "8d04a686-d5e8-4798-a27d-c608e4d0e187@26.1"
      },
      {
        "name": "College Physics with Courseware",
        "vocabularies": [
          "domain",
          "innovation"
        ],
        "vuid": "405335a3-7cff-4df2-a9ad-29062a4af261@7.53"
      },
      {
        "name": "Introduction to Sociology 2e",
        "vocabularies": [
          "domain",
          "innovation"
        ],
        "vuid": "02040312-72c8-441e-a685-20e9333f3e1d@10.1"
      },
      {
        "name": "Biology for AP® Courses",
        "vocabularies": [
          "domain",
          "innovation"
        ],
        "vuid": "6c322e32-9fb0-4c4d-a1d7-20c95c5c7af2@18.4"
      }
    ]
  },
  "started": "Tue Oct  1 16:09:23 2019",
  "version": {
    "date": "2019-10-01T14:40:38-0500",
    "dirty": false,
    "error": null,
    "full-revisionid": "ca00a4f816dabe4e97950b81bc8f178437b105e3",
    "version": "2.3.0"
  }
}
```
The `datasets` at the top list the books that have their vocabularies loaded and available.

### Dataset APIs

The following routes all serve JSON formatted representations of the datasets used by the
validator to make its validity determinations. Currently, book vocabularies are available.
In the future, this will be expanded to the exercise vocabularies, as well as the weights used
to combine the feature values (feature coefficients).

|Route|Response
|-----|--------
/datasets | list of classes of datasets available
/datasets/books| list of books
/datasets/books/`<book-vuid>`| Data for a single book
/datasets/books/`<book-vuid>`/vocabularies | list of vocabularies for a single book
/datasets/books/`<book-vuid>`/vocabularies/domain | list of non-common words in the book
/datasets/books/`<book-vuid>`/vocabularies/innovation | lists of novel words in each page of the book, by page
/datasets/books/`<book-vuid>`/vocabularies/innovation/`<page-vuid>` | list of novel words for a specific page in the book

### Processing APIs

Route|Propose|Use
---|---|---
/import|load a book and associated exercises| POST a tutor ecosystem YAML file
/train|find best-fit feature coefficents| POST a response training set


#### TODO:

- store feature coefficent sets, return IDs
- additional data APIs for downloading exercise vocabularies and feature cofficient sets
- Currently there is no security for this app (anything can call it).  I am not sure how this is usually handled in Tutor but it should not be too difficult to add an api key or similar security measures.
- Depending on UX, we may want to return more granular information about the response rather than a simple valid/non-valid label.  We can modify this easily enough as the need arises.

