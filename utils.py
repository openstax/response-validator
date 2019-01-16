# utils.py
# Author: drew
# Load up the relevant book and question data and transform into the simplified data frames we need for garbage detection

import pandas as pd
import numpy as np
import sys
import re
from nltk.corpus import words


def create_fixed_data():


	df_grouped = pd.read_csv('./ml/data/book_dataframe.csv')
	df_questions = pd.read_csv('./ml/data/question_dataframe.csv')

	# Get domain-level and module-level vocabulary innovation
	# Computes words that are novel at that particular level (over general corpora)
	df_grouped['innovation_words'] = ''
	L = df_grouped.shape[0]
	cumulative_word_set = set(words.words())
	for ll in range(0, L):
	    text = df_grouped.iloc[ll].text.lower()
	    text = re.sub('[!?().,;"“”:0-9]', ' ', text)
	    current_words = set(text.split())
	    innovation_words = current_words - cumulative_word_set
	    df_grouped['innovation_words'].iloc[ll] = innovation_words
	    cumulative_word_set = cumulative_word_set | current_words

	# Final stuff
	df_innovation = df_grouped.rename(columns={'book_name': 'subject_name', 'CNX Chapter Number': 'chapter_id', 'CNX Section Number': 'section_id'})
	df_domain = pd.DataFrame({'subject_name': 'Biology', 'domain_words': set.union(*df_innovation.innovation_words.values.tolist())}).iloc[0:1]

	return (df_innovation, df_domain, df_questions)
