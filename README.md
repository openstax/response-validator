# response_validation_app

Implements a simple unsupervised method for classifying student short to medium sized responses to questions.

## Installation

This was developed in Python 3.6. 

It may be installed as a package from the pypi repository, using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install response-validator
```

You will also need to download the corpora for the NLTK package.  This needs to
be done as the user that will run the server, inside the python environment that
will be used:

```bash
python -m nltk.downloader snowball_data words stopwords punkt
```

or as another simpler command: (at the cost of additional disk space)

```bash
python -m nltk.downloader all
```

should accomplish this.


## Development
After cloning the repository, you can install the repo in editable mode, as so:

```bash
pip install -e .
```

Additional functionality for running algorith tests, etc. can be enabled by installing additional libraries:

```bash
pip install -r requirements.txt
```

## Usage

Once installed, `python -m validator.app` will run the Flask dev webserver.

The recommended production method for deployment is to use a WSGI compliant
server, such as gunicorn:

```bash
pip install gunicorn
gunicorn validator.app:app
```

## API

The main route for the app is /validate, which accepts a plaintext response (`response`) that will be checked.  It can also accept a number of optional arguments:

- `uid` (e.g., '1000@1', default None): This is the uid for the question pertaining to the response. The uid is used to compute domain-specific and module-specific vocabulary to aid in the classification process.
Iff the version of the question specified is not available, any version of the same qid (question id without the version, e.g. 1000) will be used. 

- `remove_stopwords` (True or False, default True): Whether or not stopwords (e.g., 'the', 'and', etc) will be removed from the response.  This is generally advised since these words carry little predictive value.

- `tag_numeric` (True, False or auto, default auto): Whether numerical values will be tagged (e.g., 123.7 is tagged with a special 'numeric_type_float' identifier). While there are certainly responses for which this would be helpful, a large amount of student garbage consists of random number pressing which limits the utility of this option. Auto enables a mode that only does numeric tag processing if the question this response pertains to (as fond via the uid above) requires a numeric answer.

- `spelling_correction` (True, False or auto, default auto): Whether the app will attempt to correct misspellings. This is done by identifying unknown words in the response and seeing if a closely related known word can be substituted.  Currently, the app only attempts spelling correction on words of at least 5 characters in length and only considers candidate words that are within an edit distance of 2 from the misspelled word. When running in `auto` mode, the app will attempt to determine validity without spelling correction. Only if that is not valid, will it attempt to reasses validty with spelling correction.

- `spell_correction_max` (integer, default 10): Limit spelling corrections applied to this number.

- `remove_nonwords` (True or False, default True): Words that are not recognized (after possibly attempting spelling correction) are flagged with a special 'nonsense_word' tag.  This is done primarily to combat keyboard mashes (e.g., 'asdfljasdfk') that make a large percentage of invalid student responses.

Once the app is running, you can send requests using curl, requests, etc.  Here is an example using Python's requests library:

Here an example of how to call things using the Python requests library (assuming that the app is running on the default local development port):

```python
import json
import requests
params = {'response': 'this is my answer to the question alkjsdfl',
          'uid': '100@2',
          'remove_stopwords': True,
          'tag_numeric=True': False,
          'spelling_correction': True,
          'remove_nonwords': True}
r = requests.get('http://127.0.0.1:5000/validate', params=params)
print(json.dumps(r.json(), indent=2))
{
  "bad_word_count": 1,
  "common_word_count": 2,
  "computation_time": 0.001367330551147461,
  "domain_word_count": 0,
  "inner_product": -1.6,
  "innovation_word_count": 0,
  "num_spelling_correction": 1,
  "processed_response": "answer question nonsense_word",
  "remove_nonwords": true,
  "remove_stopwords": true,
  "response": "this is my answer to the question alkjsdfl",
  "spelling_correction": true,
  "spelling_correction_used": true,
  "tag_numeric": "auto",
  "tag_numeric_input": "auto",
  "uid_found": false,
  "uid_used": null,
  "valid": false
}


```

## TODO:

While the app is fully functional, there are some other things that will need to be addressed:

- Currently there is no security for this app (anything can call it).  I am not sure how this is usually handled in Tutor but it should not be too difficult to add an api key or similar security measures.
- The Procfile will need to be changed a bit depending on how and where we wish to deploy
- By far the largest element of the processing time for a response is devoted to spelling correction. While this does provide a very strong performance improvement for short responses, we may wish to automatically disable this in the case where the response is too long (larger than a paragraph).
- Depending on UX, we may want to return more granular information about the response rather than a simple valid/non-valid label.  We can modify this easily enough as the need arises.
