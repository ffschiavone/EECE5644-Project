import numpy as np
import pandas as pd


def generate_shuffled_data():
    df = pd.read_csv('./cleaned_data.csv', index_col='Respondent')
    shuffled_df = df.loc[np.random.permutation(df.index)]
    shuffled_df.to_csv(path_or_buf='./shuffled.csv')


if __name__ == '__main__':
    generate_shuffled_data()

