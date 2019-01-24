# unsupervised_garbage_detection.py
# Created by: Drew
# This file implements the unsupervised garabage detection variants and simulates accuracy/complexity tradeoffs

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from utils import create_fixed_data
from ml.stax_string_proc import StaxStringProc
from nltk.corpus import words
import re
import time

app = Flask(__name__)

# Default parameters for the response parser
remove_stopwords_default = True
tag_numeric_default = False
spelling_correction_default = True
remove_nonwords_default = True
weights = np.array([-3, 2.5, 2.2, .7])

# Get the global data for the app (innovation words by module, domain words by subject, and table linking question uid to cnxmod)
print("Loading up the data . . . this may take a bit")
df_innovation, df_domain, df_questions = create_fixed_data()
print("Finished!")
question_set = df_questions.uid.values.tolist()

# Define common and bad vocab
# Define common and bad vocab
with open('./ml/corpora/bad.txt') as f:
	bad_vocab = set([re.sub('\n', '', w) for w in f])

# Create the parser, initially assign default values (these can be overwritten during calls to process_string)
parser = StaxStringProc(corpora_list=['./ml/corpora/big.txt',
		   							  './ml/corpora/all_plaintext.txt'],
                        parse_args=(remove_stopwords_default, tag_numeric_default, spelling_correction_default, remove_nonwords_default))

common_vocab = set(words.words()) | set(parser.reserved_tags)


# Function to estimate validity given response, uid, and parser parameters
def validate_response(response, uid, remove_stopwords, tag_numeric, spelling_correction, remove_nonwords):
	
	# Get innovation and domain vocabulary
	# Requires a valid UID - otherwise will just use empty sets for these
	innovation_vocab = set()
	domain_vocab = set()
	if (uid in question_set):
		module_id = df_questions[df_questions['uid']==uid].iloc[0].module_id
		innovation_vocab = df_innovation[df_innovation['module_id']==module_id].iloc[0].innovation_words
		subject_name = df_innovation[df_innovation['module_id']==module_id].iloc[0].subject_name
		domain_vocab = df_domain[df_domain['subject_name']==subject_name].iloc[0].domain_words

	# Parse the students response into a word list
	response_word_list = parser.process_string(response,
											   remove_stopwords=remove_stopwords,
                       						   tag_numeric=tag_numeric,
                       						   correct_spelling=spelling_correction,
                       						   kill_nonwords=remove_nonwords)

	# Compute intersection cardinality with each of the sets of interest
	bad_word_count = sum([w in bad_vocab for w in response_word_list])
	domain_word_count = sum([w in domain_vocab for w in response_word_list])
	innovation_word_count = sum([w in innovation_vocab for w in response_word_list])
	common_word_count = sum([w in common_vocab for w in response_word_list])

	# Group the counts together, compute an inner product with the weights, and return the results
	vector = np.array([bad_word_count, domain_word_count, innovation_word_count, common_word_count])
	inner_product = np.sum(vector*weights)

	return {'response': response,
			'remove_stopwords': remove_stopwords,
			'tag_numeric': tag_numeric,
			'spelling_correction': spelling_correction,
			'remove_nonwords': remove_nonwords,
			'processed_response': " ".join(response_word_list),
			'uid': uid,
			'uid_found': uid in question_set,
			'bad_word_count': bad_word_count,
			'domain_word_count': domain_word_count,
			'innovation_word_count': innovation_word_count,
			'common_word_count': common_word_count,
			'inner_product': inner_product,
			'valid': str(inner_product>0)
			}

# Defines the entry point for the api call
# Read in/preps the validity arguments and then calls validate_response
# Returns JSON dictionary
@app.route('/validate')
def validation_api_entry():

	start_time = time.time()

	# Get the route arguments . . . use defaults if not supplied
	response = request.args.get('response', None)
	uid = request.args.get('uid', None)
	remove_stopwords = request.args.get('remove_stopwords', remove_stopwords_default) == 'True'
	tag_numeric = request.args.get('tag_numeric', tag_numeric_default) == 'True'
	spelling_correction = request.args.get('spelling_correction', spelling_correction_default) == 'True'
	remove_nonwords = request.args.get('remove_nonwords', remove_nonwords_default) == 'True'

	return_dictionary = validate_response(response, uid, remove_stopwords, tag_numeric, spelling_correction, remove_nonwords)

	computation_time = time.time() - start_time

	return_dictionary['computation_time'] = computation_time

	return jsonify(return_dictionary)

	
if __name__ == '__main__':
    app.run(debug=False)
