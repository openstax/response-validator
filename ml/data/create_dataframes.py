# Script to process everything

import pandas as pd
import numpy as np
import sys
import re
sys.path.append('./ml/')
from nltk.corpus import words
from stax_string_proc import StaxStringProc
from nltk.tokenize import word_tokenize

###BOOK STUFF###
# Load up the books, get chapter/section info from the books export, and merge text together into single module blocks
df_bio = pd.read_csv('../ml/corpora/all_the_books.csv')
df_bio = df_bio.dropna(subset=['text'])
df_bio_summary = df_bio[df_bio['tag']!='figure']
df_bio_summary = df_bio_summary[['book_name', 'id', 'module_id', 'text', 'tag']].drop_duplicates()

# Load up the book data
df_book_export = pd.read_csv('/Users/drew/Box Sync/Research/Exports/export_20181024T164046Z/cnx_export_20181024T164046Z.csv')
df_book_export = df_book_export[['CNX HTML URL', 'CNX Book Name', 'CNX Chapter Number', 'CNX Section Number', 'CNX Section Name']].drop_duplicates().reset_index()
df_book_export['cnxmod'] = df_book_export['CNX HTML URL'].apply(lambda x: x.split('/')[-1].split('@')[0])
df_book_export = df_book_export.drop_duplicates(subset=['cnxmod'])
df_book_export = df_book_export[['CNX Book Name', 'CNX Chapter Number', 'CNX Section Number', 'cnxmod']].drop_duplicates()
df_book_export = df_book_export.rename(columns={'cnxmod': 'module_id'})

# Merge and group to text by module and write to disk
df_final = df_bio_summary.merge(df_book_export, how='left')
df_final = df_final.sort_values(by=['CNX Chapter Number', 'CNX Section Number'], ascending=True)
df_grouped = df_final.groupby(['book_name', 'CNX Chapter Number', 'CNX Section Number', 'module_id'])['text'].apply(lambda x: " ".join(x)).reset_index()
df_grouped.to_csv('book_dataframe.csv', index=None)

###QUESTION STUFF###
# Load up the questions and put into right format
box_path = '~/Box Sync/Research/Data/openform_app/'
response_data_path = box_path + 'response_data.csv'
response_df = pd.read_csv(response_data_path)
question_df = response_df.drop_duplicates(subset=['uid'])[['subject_name', 'uid', 'chapter_id', 'section_id']]
question_df = question_df.merge(df_grouped[['module_id', 'CNX Chapter Number', 'CNX Section Number']],
                               left_on=['chapter_id', 'section_id'], right_on=['CNX Chapter Number', 'CNX Section Number'],
                               how='left')
question_df = question_df[['subject_name', 'uid', 'chapter_id','section_id', 'module_id']].drop_duplicates(subset=['uid'])
question_df = question_df.dropna(subset=['module_id'])
question_df.to_csv('question_dataframe.csv', index=None)


