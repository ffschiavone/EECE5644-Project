# Import the necessary modules and libraries
import numpy as np
from sklearn.neighbors.regression  import KNeighborsRegressor
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)


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
    x, y = to_xy(df, solve_column)

    return train_test_split(x, y, test_size=test_size)


def run_knn_regression(column_weights, solve_column, test_size):

    x_train, x_test, y_train, y_test = get_data(column_weights, solve_column, test_size)
    n_samples = len(x_train)

    n_neighbors = 10
    if n_samples == 0:
        return column_weights.keys(), 1000000
    elif n_samples < n_neighbors:
        n_neighbors = n_samples

    regr = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
    regr.fit(x_train, y_train)
    predictions = regr.predict(x_test)
    errors = []
    for index in range(len(predictions)):
        errors.append(predictions[index] - y_test[index][0])

    #plot histogram
    #plt.hist(np.vstack(errors), bins=50, histtype='step')
    #plt.show()
    score = regr.score(x_test, y_test)
    rms = sqrt(mean_squared_error(y_test, predictions))
    return column_weights.keys(), rms, score

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
    return sorted(indicators, key = lambda x: x[1])

if __name__ == '__main__':
    column_weights = dict()
    #column_weights['YearsCodedJobPast'] = 1
    column_weights['YearsCodedJob'] = 1
    column_weights['Country'] = 11
    column_weights['YearsProgram'] = 3

    for x in range(0, 10):
        columns, rms, score = run_knn_regression(column_weights, 'Salary', 0.2)
        print(rms, score)
    #print(rank_indicators('Salary', 0.2))