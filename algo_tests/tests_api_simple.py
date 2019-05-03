# tests_api.py
# Author: drew
# Measures the tradeoffs in timing/accuracy for different configurations of the response parser and different levels of complexity in how we count vocabulary

import pandas as pd
import numpy as np
import time
# from plotnine import *
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import requests

# Users params -- where to get the response data and how much of it to sample for testing
BASE_URL = 'http://127.0.0.1:5000/validate' #Local Path 'https://protected-earth-88152.herokuapp.com/validate' #Heroku path
STOPS = [True]
NUMS = ['auto', False, True]
SPELL = [True]
NONWORDS = [True]
USE_UID = [True]
DATAPATHS = ['./data/expert_grader_valid_100.csv', './data/alicia_valid.csv']

# Simple helper function to process the result of the api call into something nice for a pandas dataframe
def do_api_time_call(response, uid, stops, nums, spell, nonwords, use_uid):
	if (not use_uid):
		uid = None
	params = {'response': response, 'uid': uid, 'remove_stopwords': stops, 'tag_numeric': nums, 'spelling_correction': spell, 'remove_nonwords': nonwords}
	r = requests.get(BASE_URL, params=params)
	D = r.json()
	strings = [str(D[k]) for k in D.keys()]
	return "xxx".join(strings)


# Iterate through all parser/vocab combinations and get average timing estimates per response
# Then do a 5-fold cross validation to estimate accuracy
print("Starting the test")

df_results = pd.DataFrame()

for datapath, stops, nums, spell, nonwords, use_uid in product(DATAPATHS, STOPS, NUMS, SPELL, NONWORDS, USE_UID):

	# Load the data
	dft = pd.read_csv(datapath)
	dft['data'] = datapath
	n_samp = dft.shape[0]

	# Compute the actual features and do the timing computation (normalized per response)
	now = time.time()
	dft['result'] = dft.apply(lambda x: do_api_time_call(x.free_response, x.uid, stops, nums, spell, nonwords, use_uid), axis=1)
	elapsed_time_total = (time.time()-now)
	elapsed_time = elapsed_time_total / n_samp

	dft_results = dft.result.str.split('xxx', expand=True)
	dft_results.columns = ['bad_word_count', 'common_word_count', 'computation_time', 'domain_word_count', 'inner_product',
					'innovation_word_count', 'processed_response', 'remove_nonwords', 'remove_stopwords', 'response',
					'spelling_correction', 'tag_numeric', 'tag_numeric_input', 'uid_used', 'uid_found', 'valid_result']
	dft_results = pd.concat([dft, dft_results], axis=1)
	dft_results['computation_time'] = dft_results['computation_time'].astype(float)
	dft_results['pred_correct'] = dft_results['valid_result'].map({'True': True, 'False': False}) == dft_results['valid']
	df_results = df_results.append(dft_results)

# Compile and display some results
# 1) Plot accuracy by numerical parsing scheme
res = df_results.groupby(['data', 'tag_numeric_input'])['pred_correct'].mean().reset_index()
# acc_plot = ggplot(res, aes('tag_numeric_input', 'pred_correct')) + geom_bar(stat='identity') + facet_wrap('~data') + xlab('Numerical Parsing Scheme') + ylab('Accuracy')
print(res)

# 2 For Alicia's data, pivot so that we can see where different cases are failing by uid
# Ultimately want auto to agree with True for the most part 
pivot_results = df_results[df_results['data']=='./data/alicia_valid.csv'].pivot(index='uid', columns='tag_numeric_input', values='pred_correct').reset_index()
print(pivot_results)

