# Created by aew2
# Script will take in a path to a Tutor Exercise export and will create the file
# df_questions.csv used by response-validator with the following columns:
# ['uid', 'text', 'stem_text', 'option_text', 'contains_number', 'module_id', 'CNX Book Name', 'CNX Chapter Number', 'CNX Section Number']

import pandas as pd
import sys
sys.path.append('..')
from stax_string_proc import StaxStringProc
import re

path_to_export = '/Users/drew/Box Sync/Research/External/NSF Highlight Share/2018 Data Complete/'
exercise_filename = 'exercises_export_20190122T144506Z.csv'
book_filename = 'cnx_export_20190122T144506Z.csv'
tutor_filename = 'tutor_export_20190122T144506Z.csv'
max_options = 6

# Helper function to check for embedded math and remove HTML-style tags (crudely)
def clean_text(x):
    x = re.sub('<span data-math="', '', x)
    x = re.sub('"></span>', '', x)
    x = re.sub("<[^>]*>", "", x)
    x = re.sub('\n', '', x)
    return x

df = pd.read_csv(path_to_export + exercise_filename)
df_book = pd.read_csv(path_to_export + book_filename)
df_tutor = pd.read_csv(path_to_export + tutor_filename)

# Create the UID from the Exercise Number and Exercise Version
df['uid'] = df.apply(lambda x: str(x['Exercise Number']) + '@' + str(x['Exercise Version']),
                     axis=1)

# Get the stem text
df['stem_text'] = df['Question Stem HTML'].apply(lambda x: clean_text(x))

# Merge all non-empty options together
L = ['Answer ' + str(i) + ' Content HTML' for i in range(1, max_options + 1)]
for l in L:
    df[l] = df[l].fillna('')
    df[l] = df[l].apply(lambda x: clean_text(x))
df['option_text'] = df[L].apply(lambda x: ' '.join(x.dropna()), axis=1)

# text field is combination of stem and options
df['text'] = df[['stem_text', 'option_text']].apply(lambda x: ' '.join(x.dropna()), axis=1)

# Do numeric tagging on everything
def has_num(x, parser, tag_set):
    token_list = sp.process_string_spelling_limit(x, tag_numeric=True)[0]
    if len(set(token_list) & set(tag_set)) > 0:
        return True
    else:
        return False

sp = StaxStringProc(corpora_list=["../corpora/all_plaintext.txt",
                                  "../corpora/big.txt",
                                  ])
tag_set = [x for x in sp.reserved_tags if ('numeric' in x) or ('math' in x)]
df['contains_number'] = df['text'].apply(lambda x: has_num(x, sp, tag_set))

# Merge in the module and book information
df_tutor = df_tutor[['Exercise Editor URL', 'CNX HTML URL']].dropna().drop_duplicates()
df_tutor['uid'] = df_tutor['Exercise Editor URL'].apply(lambda x: x.split('/')[-1])
df_tutor = df_tutor.merge(df_book[['CNX HTML URL', 'CNX Book Name', 'CNX Chapter Number', 'CNX Section Number']],
                          how='left')
df_tutor['module_id'] = df_tutor['CNX HTML URL'].apply(lambda x: x.split('/')[-1])
df_tutor = df_tutor[['uid', 'CNX Book Name', 'CNX Chapter Number', 'CNX Section Number', 'module_id']]
df_tutor = df_tutor.drop_duplicates(subset=['uid'])

# Do the final merge, select, and write
df = df.merge(df_tutor, on=['uid'], how='left')
df = df[['uid', 'text', 'stem_text', 'option_text', 'contains_number', 'module_id', 'CNX Book Name', 'CNX Chapter Number', 'CNX Section Number']]

df.to_csv('df_questions.csv')










