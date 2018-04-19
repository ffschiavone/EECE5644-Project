# Import the necessary modules and libraries
import numpy as np
from sklearn.neighbors.regression  import KNeighborsRegressor
import pandas as pd
import json
from math import sqrt
from sklearn.metrics import mean_squared_error


def reduce_columns(df, columns):
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)
    # def keep_columns(df, original_columns_to_keep):
    cleaned_columns_to_keep = []
    for original_col in columns:
        cleaned_columns_to_keep += column_mapping[original_col]
    return df[cleaned_columns_to_keep].dropna()

def add_column_weights(df, columns_weights):
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)

    for column, weight in columns_weights.items():
        cleaned_columns = column_mapping[column]
        df[cleaned_columns] *= weight
    return df


def parse_csv():
    df = pd.read_csv('../cleaned_data.csv')

    return df

def clean_data(df, column_weights, solve_column):
    columns = list(column_weights.keys())
    columns.append(solve_column)
    df = reduce_columns(df, columns)
    df = add_column_weights(df, column_weights)
    return df

def get_data(column_weights, solve_column, test_size):
    df = parse_csv()
    df = clean_data(df, column_weights, solve_column)

    return df


def run_knn_regression(column_weights, solve_column, test_size):
    df = get_data(column_weights, solve_column, test_size)
    shuffled_df = df.loc[np.random.permutation(df.index)]
    mse_sum = 0
    num_train_set = 10
    for i in range(num_train_set):
        test = shuffled_df.iloc[int(len(df) * i / num_train_set): int(len(df) * (i + 1) / num_train_set)]
        test_without_salary = test.drop(columns='Salary')
        train = df.drop(test.index)
        train_without_salary = train.drop(columns='Salary')

        regr = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
        regr.fit(train_without_salary, train['Salary'])
        predictions = regr.predict(test_without_salary)
        mse_sum += sqrt(mean_squared_error(test['Salary'], predictions))

    #plot histogram
    #plt.hist(np.vstack(errors), bins=50, histtype='step')
    #plt.show()
    #score = regr.score(x_test, y_test)
    #rms = sqrt(mean_squared_error(y_test, predictions))
    return column_weights.keys(), mse_sum/10

def rank_indicators(solve_column, test_size):
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)
    keys = list(column_mapping.keys())
    keys.remove(solve_column)
    keys.remove('ExpectedSalary')
    indicators = list()
    print(keys)
    print(len(keys))
    for key in keys:
        column, rms = run_knn_regression({key: 1.0}, solve_column, test_size)
        indicators.append([key, rms])
        print(column, rms)
    return sorted(indicators, key=lambda x: x[1])

if __name__ == '__main__':
    column_weights = dict()
    column_weights['YearsCodedJob'] = 1
    column_weights['Country'] = 11
    column_weights['YearsProgram'] = 3

    columns, rms = run_knn_regression(column_weights, 'Salary', 0.2)
    print(rms)
