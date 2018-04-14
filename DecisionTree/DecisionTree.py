# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
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


def reduce_columns(df):
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)
    # Important columns: YearsProgram, YearsCodedJob, Country, ImportantBenifits, CompanyType
    original_columns_to_keep = ['YearsProgram', 'YearsCodedJob', 'Country', 'ImportantBenefits', 'CompanyType', 'Salary']
    # def keep_columns(df, original_columns_to_keep):
    cleaned_columns_to_keep = []
    for original_col in original_columns_to_keep:
        cleaned_columns_to_keep += column_mapping[original_col]
    return df[cleaned_columns_to_keep].dropna()


def parse_csv():
    df = pd.read_csv('../cleaned_data.csv')
    df = reduce_columns(df)
    return df


def get_data():
    df = parse_csv()
    x, y = to_xy(df, 'Salary')
    # x = np.delete(x, [288, 299, 290, 291, 292, 295], axis=1)
    # # iterate through all data, if there is a nan, infinite or blank value, drop the row
    # nans = np.where(np.isnan(x))
    # x = np.delete(x, nans[0], axis=0)
    # y = np.delete(y, nans[0], axis=0)

    return train_test_split(x, y, test_size=0.20)


def run_decision_tree():
    x_train, x_test, y_train, y_test = get_data()
    best_rms = 100000
    regr = DecisionTreeRegressor(max_depth=28, min_samples_split=175, min_samples_leaf=3, max_leaf_nodes=129)
    regr.fit(x_train, y_train)
    predictions = regr.predict(x_test)
    errors = []
    for index in range(len(predictions)):
        errors.append(predictions[index] - y_test[index][0])
    plt.hist(errors, bins=30, histtype='step')
    plt.show()
    score = regr.score(x_test, y_test)
    rms = sqrt(mean_squared_error(y_test, predictions))
    if rms < best_rms:
        best_rms = rms
    print('New best rms: {0} Score: {1}'.format(rms, score))


if __name__ == '__main__':
    run_decision_tree()
