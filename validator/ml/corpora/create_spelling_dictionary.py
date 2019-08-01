# This script creates a SymSpell spelling dictionary for use with the parser
# It takes in a combination of a previously existant symspell dictionary and external files for training
# The code flow is as follows:
# 1) Load up the original symspell dictionary
# 2) Create dictionaries for all of the external files (textbooks, questions banks, etc) and append in one pd frame
# 3) Adjust the counts in the original dataframe to be comparable to the external stuff (keep the prior favorable)
# 4) Merge everything together, write to disk

from symspellpy.symspellpy import SymSpell, Verbosity
import pandas as pd
import numpy as np
import re

# symspell_dictionary is a symspell ready file (plaintext, term is 0th col, count is 1st col)
# external_files are each plaintext -- counts will be computed on-the-fly
symspell_dictionary = 'frequency_dictionary_en_82_765.txt'
external_files = ['all_plaintext.txt', 'question_text.txt']
output_dictionary_name = 'response_validator_spelling_dictionary.txt'

# 1) Load up the original symspell dictionary and convert to pandas dataframe
sym_spell = SymSpell(3, 7)
sym_spell.load_dictionary(symspell_dictionary, 0, 1)
df_original = pd.DataFrame.from_dict(sym_spell.words, orient='index').reset_index()
df_original.columns = ['term', 'count']

# 2) Create a dictionary for each of the external datafiles. Append together and get total counts for each term
df_external = pd.DataFrame()
regexp = re.compile(r"[.!?\-\\+\[\]\#\$\%\^\&\*\(\)\@\d\']+")
for file in external_files:
    sym_spell = SymSpell(3, 7)
    sym_spell.create_dictionary(file)
    df_temp = pd.DataFrame.from_dict(sym_spell.words, orient='index').reset_index()
    df_temp.columns = ['term', 'count']
    df_temp = df_temp[df_temp['term'].apply(lambda x: not regexp.search(x))]
    df_external = df_external.append(df_temp)
df_external = df_external.groupby('term')['count'].sum().reset_index()

# 3) Adjust the counts in the original dataframe to be comparable to those in the external dataframe
# This avoids mangling the prior when doing Bayesian spelling correction
N_external = df_external['count'].sum()
N_original = df_original['count'].sum()
df_original['count'] = df_original['count'].apply(lambda x: int(np.ceil(x / (N_original / N_external))))

# 4) Merge everything together, get final counts, and write to disk
df_final = df_original.append(df_external)
df_final = df_final.groupby('term')['count'].sum().reset_index()
df_final = df_final[df_final['term'].apply(lambda x: len(x)>1)]
df_final.to_csv(output_dictionary_name,
                header=False,
                index=None,
                sep=' ')



