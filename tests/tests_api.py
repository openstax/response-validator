# tests_api.py
# Author: drew
# Measures the tradeoffs in timing/accuracy for different configurations of the response parser and different levels of complexity in how we count vocabulary
# Creates two dataframes:
# a) api_simuilation_results.csv: A table of 2**5 rows with average timing and accuracy data (one row for each possible configuration)
# b) api_simulation_delta.csv: A table showing the average gain in accuracy/timing for each of the five possible parameters

import pandas as pd
import numpy as np
import time
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import requests

# Users params -- where to get the response data and how much of it to sample for testing
base_url = 'http://127.0.0.1:5000/validate'
response_path = '~/Box Sync/Research/Data/openform_app/response_data.csv'
n_samp = 1000

# Read the data, set up the default LogisticRegression model
df_full = pd.read_csv(response_path)
df_full = df_full[df_full['subject_name']=='Biology']
df = df_full.sample(n=n_samp, random_state=42)
lr = LogisticRegression()

# Simple helper function to process the result of the api call into something nice for a pandas dataframe
def do_api_call(response, uid, stops, nums, spell, nonwords, use_uid):

	if (not use_uid):
		uid = None

	params = {'response': response, 'uid': uid, 'remove_stopwords': stops, 'tag_numeric': nums, 'spelling_correction': spell, 'remove_nonwords': nonwords}
	r = requests.get(base_url, params=params)
	D = r.json()
	stuff_i_want = [D['bad_word_count'], D['domain_word_count'], D['innovation_word_count'], D['common_word_count']]
	stuff_i_want = [str(e) for e in stuff_i_want]
	return ", ".join(stuff_i_want)


# Iterate through all parser/vocab combinations and get average timing estimates per response
# Then do a 5-fold cross validation to estimate accuracy
df_results = pd.DataFrame()
print("Starting the test")


# Does not use stemming or common filtering (the latter just gets caught in garbage detection so meh)
counter = 1
for stops, nums, spell, nonwords, use_uid in product([True, False], repeat=5):
	
	print('Iter: ' + str(counter) + '/' + str(2**5))
	counter = counter + 1

	# Make a local copy of the dataframe
	dft = df.copy()

	# Compute the actual features and do the timing computation (normalized per response)
	now = time.time()
	dft['vals'] = dft.apply(lambda x: do_api_call(x.free_response, x.uid, stops, nums, spell, nonwords, use_uid), axis=1)
	elapsed_time = (time.time()-now) / n_samp

	# Now, separate out the values, do classification stuff, and write result to the final dataframe
	dft['B'] = dft['vals'].apply(lambda x: int(x.split(', ')[0]))
	dft['D'] = dft['vals'].apply(lambda x: int(x.split(', ')[1]))
	dft['I'] = dft['vals'].apply(lambda x: int(x.split(', ')[2]))
	dft['C'] = dft['vals'].apply(lambda x: int(x.split(', ')[3]))
	X = dft[['B', 'D', 'I', 'C']].values
	y = dft['junk'].values
	acc = np.mean(cross_val_score(lr, X, y, cv=5))
	df_results = df_results.append(pd.DataFrame({'stopword_removal': [stops],
							 					 'numerical_tagging': [nums],
												 'spelling_correction': [spell],
		 										 'remove_nonwords': [nonwords],
												 'use_domain_vocab': [use_uid],
												 'average_time': [elapsed_time],
												 'accuracy': [acc]}))


# Write per-trial results to the disk
df_results.to_csv('api_simuilation_results.csv', index=None)

# Measure average impact of timing on accuracy for each of the 5 parameters
# This is done by looking across all combinations of the other four parameters and averaging the impact of adding the 5th
# This could probably be simplified to a few melt/pivot commands . . .
df_analysis = df_results.copy()
cols = ['stopword_removal', 'numerical_tagging', 'remove_nonwords', 'spelling_correction', 'use_domain_vocab']
df_analysis[cols] = df_analysis[cols].astype(str)
df_analysis_results = pd.DataFrame()
for col in cols:
	ctemp = [c for c in cols if c!=col]
	df_analysis['key'] = df_analysis[ctemp].apply(lambda x: "".join(x), axis=1)
	df_true = df_analysis[df_analysis[col]=='True'][['key', col, 'accuracy', 'average_time']]
	df_false = df_analysis[df_analysis[col]=='False'][['key', col, 'accuracy', 'average_time']]
	dft = pd.merge(df_true, df_false, on='key')
	dft['accuracy_delta'] = dft['accuracy_x'] - dft['accuracy_y']
	dft['time_delta'] = dft['average_time_x'] - dft['average_time_y']
	temp = dft[['accuracy_delta', 'time_delta']].agg({'mean', 'std'}).reset_index().rename(columns={'index': 'measure'})
	temp['variable'] = col
	df_analysis_results = df_analysis_results.append(temp)
df_analysis_mean = df_analysis_results[df_analysis_results['measure']=='mean'].drop('measure', axis=1)
df_analysis_mean = df_analysis_mean.rename(columns={'accuracy_delta': 'accuracy_delta__mean', 'time_delta': 'time_delta__mean'})
df_analysis_std = df_analysis_results[df_analysis_results['measure']=='std'].drop('measure', axis=1)
df_analysis_std['accuracy_delta'] = df_analysis_std['accuracy_delta'] / np.sqrt(2**len(ctemp))
df_analysis_std['time_delta'] = df_analysis_std['time_delta'] / np.sqrt(2**len(ctemp))
df_analysis_std = df_analysis_std.rename(columns={'accuracy_delta': 'accuracy_delta__std', 'time_delta': 'time_delta__std'})
df_final = pd.merge(df_analysis_mean, df_analysis_std, on='variable')
df_final = pd.melt(df_final, id_vars='variable', var_name='thing')
df_final['measure'] = df_final['thing'].apply(lambda x: x.split('__')[0])
df_final['metric'] = df_final['thing'].apply(lambda x: x.split('__')[1])
df_final = df_final.drop('thing', axis=1)
df1 = df_final[df_final['metric']=='mean'].drop('metric', axis=1).rename(columns={'value': 'mean'})
df2 = df_final[df_final['metric']=='std'].drop('metric', axis=1).rename(columns={'value': 'std'})
df_final = pd.merge(df1, df2, on=['variable', 'measure'])
df_final.to_csv('api_simulation_delta.csv', index=None)

# Create a visualization of the relative impacts -- requires plotnine library
# from plotnine import *
# ggplot(df_final, aes('variable', 'mean')) + geom_bar(stat='identity') + facet_wrap('~measure') + theme(axis_text_x = element_text(angle = 45, hjust = 1))
