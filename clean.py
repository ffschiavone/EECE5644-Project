import numpy as np
import pandas as pd

df_raw = pd.read_csv('survey_results_public.csv', index_col='Respondent')
df_raw = df_raw[df_raw['Salary'].notnull()]

multi_select_cols = []
numeric_cols = []
other_cols = []

for col in df_raw.columns:
    series = df_raw[col]
    
    if series.dtype == np.float64:
        numeric_cols.append(col)
        continue
    
    is_multi_select = False
    for entry in series.unique():
        if type(entry) == str and ';' in entry:
            multi_select_cols.append(col)
            is_multi_select = True
            break
    if is_multi_select:
        continue
    
    other_cols.append(col)

df = df_raw[other_cols + numeric_cols]
for col in df.columns:
    if col.startswith('Years'):
        years_as_string = df_raw[col] \
            .fillna('') \
            .replace('Less than a year', '0 to 1 years') \
            .str.extract('(\d+) ', expand=False)
        df[col] = pd.to_numeric(years_as_string) + 0.5

df = pd.get_dummies(df)
for multi_select_col in multi_select_cols:
    to_concat = df_raw[multi_select_col].str.get_dummies(sep='; ')
    df = pd.concat([df, to_concat], axis=1)

df.to_csv('cleaned_data.csv')