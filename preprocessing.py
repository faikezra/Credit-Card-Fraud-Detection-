# Data preprocessing for Credit-Card-Fraud-Detection problem
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def skew_amount(data):
    normal_data = data[data.Class == 0]
    anomaly_data = data[data.Class != 0]
    print('Data points that are normal: {}'.format(len(data[data.Class == 0])))
    print('Data points that are abnormal: {}'.format(len(data[data.Class != 0])))

def normalize_amount(data):
    normalizer = StandardScaler()
    data['Normalized_Amount'] = normalizer.fit_transform(np.array(data['Amount']).reshape(-1, 1))
    data = data.drop('Amount', axis = 1)
    return (data)

def stratify_split(data, anomaly_count):
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']
    normal_data = data[data.Class == 0]
    anomaly_data = data[data.Class != 0]
    normal_index = normal_data.sample(anomaly_count).index
    anomaly_index = anomaly_data.sample(anomaly_count).index
    all_index = np.concatenate([normal_index,anomaly_index])
    X_undersample = X.iloc[all_index]
    y_undersample = y.iloc[all_index]
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.1
                                                                                                   ,random_state = 0)
    X_test = X[~X.index.isin(all_index)]
    y_test = y[~y.index.isin(all_index)]
    undersample = (X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample)
    test_data = (X_test, y_test)
    print('Undersampled Train Dataset Count: {}'.format(len(y_train_undersample)))
    print('Undersampled Test Dataset Count: {}'.format(len(y_test_undersample)))
    print('Full Test Dataset Count: {}'.format(len(y_test)))

    return (undersample, test_data)

def create_datasets(dir, count):
    data = pd.read_csv(dir)
    skew_amount(data)
    data = normalize_amount(data)
    return stratify_split(data, count)
