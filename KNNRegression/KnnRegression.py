# Import the necessary modules and libraries
from sklearn.neighbors.regression import KNeighborsRegressor
import pandas as pd
import json
from math import sqrt
from sklearn.metrics import mean_squared_error


def reduce_columns(df, columns):
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)

    row_drop_mask = (df['YearsProgram'].isnull() & df['YearsCodedJob'].isnull())
    rows_to_drop = df[row_drop_mask].index
    df.drop(index=rows_to_drop, inplace=True)

    rows_with_no_YearsProgram = df[df['YearsProgram'].isnull()].index
    df.loc[rows_with_no_YearsProgram, 'YearsProgram'] = df.loc[rows_with_no_YearsProgram, 'YearsCodedJob']
    df.loc[df['YearsCodedJob'].isnull(), 'YearsCodedJob'] = 0

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
    df = pd.read_csv('../shuffled.csv')

    return df

def clean_data(df, column_weights, solve_column):
    columns = list(column_weights.keys())
    columns.append(solve_column)
    df = reduce_columns(df, columns)
    return df

def get_data(column_weights, solve_column):
    df = parse_csv()
    df = clean_data(df, column_weights, solve_column)

    return df


def run_knn_regression(column_weights, solve_column, num_train_set, num_neighbors, str_weight, p_num):
    df = get_data(column_weights, solve_column)

    df = add_column_weights(df, column_weights)

    rms_sum = 0
    for i in range(num_train_set):
        test = df.iloc[int(len(df) * i / num_train_set): int(len(df) * (i + 1) / num_train_set)]
        test_without_salary = test.drop(columns='Salary')
        train = df.drop(test.index)
        train_without_salary = train.drop(columns='Salary')

        regr = KNeighborsRegressor(n_neighbors=num_neighbors, weights=str_weight, algorithm='auto', leaf_size=30, p=p_num, metric='minkowski', metric_params=None, n_jobs=1)
        regr.fit(train_without_salary, train['Salary'])
        predictions = regr.predict(test_without_salary)
        rms_sum += sqrt(mean_squared_error(test['Salary'], predictions))

    return column_weights.keys(), rms_sum/num_train_set


def rank_indicators(solve_column, num_train_set):
    with open('../column_mapping.json', 'r') as f:
        column_mapping = json.load(f)
    keys = list(column_mapping.keys())
    keys.remove(solve_column)
    keys.remove('ExpectedSalary')
    keys.remove('YearsCodedJobPast')
    indicators = list()
    for key in keys:
        column, rms = run_knn_regression({key: 1.0}, solve_column, num_train_set, 5, 'distance', 2)
        indicators.append([key, rms])
        #print(column, rms)
    return sorted(indicators, key=lambda x: x[1])


if __name__ == '__main__':
    #print(rank_indicators('Salary', 10))
    #column_weights = dict()
    #column_weights['Country'] = 1.0

    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Country: ' + str(rms))

    #column_weights = dict()
    #column_weights['Country'] = 1.0
    #column_weights['Currency'] = 1.0

    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Country, Currency: ' + str(rms))

    #column_weights = dict()
    #column_weights['Country'] = 1.0
    #column_weights['YearsCodedJob'] = 1.0

    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Country, YearsCodedJob: ' + str(rms))

    #column_weights = dict()
    #column_weights['Country'] = 1.0
    #column_weights['YearsCodedJob'] = 1.0
    #column_weights['YearsProgram'] = 1.0

    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Country, YearsCodedJob, YearsProgram: ' + str(rms))

    #column_weights = dict()
    #column_weights['Country'] = 1.0
    #column_weights['YearsCodedJob'] = 1.0
    #column_weights['University'] = 1.0

    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Country, YearsCodedJob, University: ' + str(rms))

    column_weights = dict()
    column_weights['Country'] = 1.0
    column_weights['YearsCodedJob'] = 1.0
    column_weights['University'] = 1.0
    column_weights['TimeAfterBootcamp'] = 1.0
    columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    print('Country, YearsCodedJob, University, TimeAfterBootcamp: ' + str(rms))
    
    #Weights 5, 1, 1
    #column_weights = dict()
    #column_weights['Country'] = 5.0
    #column_weights['YearsCodedJob'] = 1.0
    #column_weights['University'] = 1.0
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Added weights 5, 1, 1: ' + str(rms))
    
    #Weights 10, 1, 1
    #column_weights = dict()
    #column_weights['Country'] = 10.0
    #column_weights['YearsCodedJob'] = 1.0
    #column_weights['University'] = 1.0
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Added weights 10, 1, 1: ' + str(rms))
    
    #Weights 15, 1.0, 1.0
    #column_weights = dict()
    #column_weights['Country'] = 15.0
    #column_weights['YearsCodedJob'] = 1.0
    #column_weights['University'] = 1.0
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Added weights 15, 1, 1: ' + str(rms))
    
    #Weights 15, 5, 1
    #column_weights = dict()
    #column_weights['Country'] = 15.0
    #column_weights['YearsCodedJob'] = 5.0
    #column_weights['University'] = 1.0
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Added weights 15, 5, 1: ' + str(rms))
    
    #Weights 15, 2, 1
    #column_weights = dict()
    #column_weights['Country'] = 15.0
    #column_weights['YearsCodedJob'] = 2.0
    #column_weights['University'] = 1.0
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('Added weights 15, 2, 1: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 1, 'uniform', 2)
    #print('k=1: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 5, 'uniform', 2)
    #print('k=5: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 10, 'uniform', 2)
    #print('k=10: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 50, 'uniform', 2)
    #print('k=50: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 100, 'uniform', 2)
    #print('k=100: ' + str(rms))
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 25, 'uniform', 2)
    #print('k=25: ' + str(rms))
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 26, 'uniform', 2)
    #print('k=26: ' + str(rms))
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 24, 'uniform', 2)
    #print('k=24: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 25, 'uniform', 1)
    #print('Distance is Manhattan ' + str(rms))
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 25, 'uniform', 2)
    #print('Distance is Euclidean: ' + str(rms))
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 25, 'uniform', 3)
    #print('Distance is Minkowski p=3: ' + str(rms))
    
    #columns, rms = run_knn_regression(column_weights, 'Salary', 10, 25, 'distance', 2)
    #print('Distance instead of uniform: ' + str(rms))
    columns, rms = run_knn_regression(column_weights, 'Salary', 5, 25, 'distance', 2)
    print('Distance instead of uniform: ' + str(rms))