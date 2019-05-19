import preprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report

def Logistic_Regression_Selection(sample_times, undersample_amount, data_dir):
    c_bank = [0.001, 0.01, 0.1, 1, 10]
    penalties = ['l1', 'l2']
    results_matrix_train = np.zeros((5,3))
    results_matrix_validation = np.zeros((5,3))
    results_matrix_large_test = np.zeros((5,3))

    for sample_count in range(sample_times):
        undersample , test_set = preprocessing.create_datasets(data_dir, undersample_amount)
        X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = undersample
        X_test, y_test = test_set
        for c in c_bank:
            for regulizer in penalties:
                log_reg = LogisticRegression(C = c, penalty= regulizer)
                log_reg.fit(X_train_undersample, y_train_undersample.values.ravel())
                # train set
                y_pred_undersample = log_reg.predict(X_train_undersample)
                recall_train = np.round(recall_score(y_train_undersample.values,y_pred_undersample), decimals=4)
                # validation set
                y_pred_undersample = log_reg.predict(X_test_undersample)
                recall_test = np.round(recall_score(y_test_undersample.values,y_pred_undersample), decimals=4)
                # large test
                y_pred_undersample = log_reg.predict(X_test)
                recall_large_test = np.round(recall_score(y_test.values,y_pred_undersample), decimals=4)
                print("------------------------------------")
                print('Sample Number {}: C-value {}, Regularizer {}, has Training Recall: {}'.format(sample_count, c, regulizer, recall_train))
                print('Sample Number {}: C-value {}, Regularizer {}, has Validation Recall: {}'.format(sample_count, c, regulizer, recall_test))
                print('Sample Number {}: C-value {}, Regularizer {}, has Validation Recall: {}'.format(sample_count, c, regulizer, recall_large_test))
                print("------------------------------------")
                results_matrix_train[c_bank.index(c)][penalties.index(regulizer)] += recall_train
                results_matrix_validation[c_bank.index(c)][penalties.index(regulizer)] += recall_test
                results_matrix_large_test[c_bank.index(c)][penalties.index(regulizer)] += recall_large_test

    results_matrix_train = results_matrix_train / sample_times
    results_matrix_validation = results_matrix_validation / sample_times
    results_matrix_large_test = results_matrix_large_test / sample_times

    result = np.where(results_matrix_train == np.amax(results_matrix_train))
    print("------------------------------------")
    print("Best Average Training Logistic Regression Recall: {}".format(np.amax(results_matrix_train)))
    print("With C value: {}".format(c_bank[result[0][0]]))
    print("With Penalty type: {}".format(penalties[result[1][0]]))
    print("------------------------------------")

    result = np.where(results_matrix_validation == np.amax(results_matrix_validation))
    print("------------------------------------")
    print("Best Average Validation Logistic Regression Recall: {}".format(np.amax(results_matrix_validation)))
    print("With C value: {}".format(c_bank[result[0][0]]))
    print("With Penalty type: {}".format(penalties[result[1][0]]))
    print("------------------------------------")

    result = np.where(results_matrix_large_test == np.amax(results_matrix_large_test))
    print("------------------------------------")
    print("Best Average Training Logistic Regression Recall: {}".format(np.amax(results_matrix_large_test)))
    print("With C value: {}".format(c_bank[result[0][0]]))
    print("With Penalty type: {}".format(penalties[result[1][0]]))
    print("------------------------------------")


sample_times = 10
undersample_amount = 200
data_dir = '/Users/ezra/Documents/data_repo/creditcard.csv'
Logistic_Regression_Selection(sample_times,undersample_amount,data_dir)
