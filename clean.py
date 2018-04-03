import json
import itertools
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# Read original data
df = pd.read_csv('survey_results_public.csv', index_col='Respondent')
df = df[df['Salary'].notnull()]

# Split columns by entry format
def is_multi_select(series):
    for entry in series.unique():
        if type(entry) == str and ';' in entry:
            return True
    return False

multi_select_cols = []
numeric_cols = []
other_cols = []

for col in df.columns:
    series = df[col]
    
    if series.dtype == np.float64:
        numeric_cols.append(col)
        continue
    
    if is_multi_select(series):
        multi_select_cols.append(col)
        continue
    
    other_cols.append(col)

# Generate column mapping
cleaned_columns_by_original_column = defaultdict(list)

for col in multi_select_cols:
    row_entries = df[col].dropna().str.split('; ').tolist()
    options = set(itertools.chain(*row_entries))
    for option in options:
        cleaned_columns_by_original_column[col].append(option)

for col in other_cols:
    if col.startswith('Years'):
        cleaned_columns_by_original_column[col].append(col)
    else:
        options = df[col].dropna().unique()
        for option in options:
            cleaned_column = col + '_' + option
            cleaned_columns_by_original_column[col].append(cleaned_column)

for col in numeric_cols:
    cleaned_columns_by_original_column[col].append(col)

with open('column_mapping.json', 'w') as f:
    json.dump(dict(cleaned_columns_by_original_column), f)


# Generate cleaned data
def clean(df):
    result = df[numeric_cols + other_cols].copy()
    
    for col in result.columns:
        if col.startswith('Years'):
            years_as_string = result[col] \
                .fillna('') \
                .replace('Less than a year', '0 to 1 years') \
                .str.extract('(\d+) ', expand=False)
            result[col] = pd.to_numeric(years_as_string) + 0.5
    
    result = pd.get_dummies(result)
    sub_dfs = []
    for multi_select_col in multi_select_cols:
        sub_df = df[multi_select_col].str.get_dummies(sep='; ')
        sub_dfs.append(sub_df)
    
    return pd.concat(sub_dfs + [result], axis=1)

clean(df).to_csv('cleaned_data.csv')	