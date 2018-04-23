# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import tree
import graphviz


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


def get_feature_keys():
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)
    # Important columns: YearsProgram, YearsCodedJob, Country, ImportantBenifits, CompanyType
    original_columns_to_keep = ['YearsProgram', 'YearsCodedJob', 'Country', 'ImportantBenefits', 'CompanyType']
    cleaned_columns_to_keep = []
    for original_col in original_columns_to_keep:
        cleaned_columns_to_keep += column_mapping[original_col]
    return cleaned_columns_to_keep


def parse_csv():
    df = pd.read_csv('../shuffled.csv')
    df = reduce_columns(df)
    return df


def get_data():
    return parse_csv()
    # x, y = to_xy(df, 'Salary')
    # x = np.delete(x, [288, 299, 290, 291, 292, 295], axis=1)
    # # iterate through all data, if there is a nan, infinite or blank value, drop the row
    # nans = np.where(np.isnan(x))
    # x = np.delete(x, nans[0], axis=0)
    # y = np.delete(y, nans[0], axis=0)

    # return train_test_split(x, y, test_size=0.20)


def run_decision_tree():
    avg_rms = 0
    # x_train, x_test, y_train, y_test = get_data()
    shuffled_df = get_data()
    # best_rms = 100000
    # avgs = []
    # for count in range(15, 30, 2):
    #     avgs.append(0)
    #     for avg_count in range(5):
    num_train_set = 10
    for i in range(num_train_set):
        test = shuffled_df.iloc[int(len(shuffled_df) * i / num_train_set): int(len(shuffled_df) * (i + 1) / num_train_set)]
        test_without_salary = test.drop(columns='Salary')
        train = shuffled_df.drop(test.index)
        train_without_salary = train.drop(columns='Salary')
        # min_samples_split=178, min_samples_leaf=3, max_leaf_nodes=129
        regr = DecisionTreeRegressor(min_samples_split=178, min_samples_leaf=3, max_leaf_nodes=129)
        regr.fit(train_without_salary, train['Salary'])
        predictions = regr.predict(test_without_salary)
        # errors = []
        # for index in range(len(predictions)):
        #     try:
        #         errors.append(predictions[index] - test['Salary'][index])
        #     except KeyError:
        #         pass
        #         # If we get a key error then we can't compare the prediction and error so we should just continue
        # plt.hist(errors, bins=30, histtype='step')
        # plt.show()
        score = regr.score(test_without_salary, test['Salary'])
        rms = sqrt(mean_squared_error(test['Salary'], predictions))

        print('New best rms RMS: {0} Score: {1}'.format(rms, score))
        avg_rms += rms
        if i > 0:
            avg_rms = avg_rms / 2
    print('Average RMS: {0}'.format(avg_rms))

    # dot_data = tree.export_graphviz(regr, out_file=None, feature_names=get_feature_keys(), class_names=['Salary'],
    #                                 filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("Decision Based Regression Tree")


if __name__ == '__main__':
    run_decision_tree()
