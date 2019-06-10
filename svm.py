# Data preprocessing for Credit-Card-Fraud-Detection problem
import preprocessing
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.cross_validation import train_test_split

def create_nonlinear_SVC(data_dir, normal_count, anomaly_count):
    undersample, _ = preprocessing.create_datasets(data_dir, normal_count, anomaly_count)
    train_x, test_x, train_y, test_y = undersample
    kernels = [ 'linear', 'poly', 'rbf', 'sigmoid']
    degree = [1,3,6,12]
    C = [1,10,20]

    recall_val = 0
    opt_kernel = None
    opt_degree = None
    opt_C = None
    for kernel in kernels:
        for deg in degree:
            for c in C:
                clf = SVC(kernel=kernel, degree = deg, C = c)
                clf.fit(train_x, train_y)
                recall_state = recall_score(test_y, clf.predict(test_x), pos_label=1)
                if recall_val < recall_state:
                    recall_val = recall_state
                    opt_kernel, opt_degree, opt_C = (kernel, deg, c)
                #print('Recall Score: {} | kernel {}, degree {}, C {}'.format(recall_state, kernel, deg, c))
    return((kernel, deg, c))

def main():
    (kernel, deg, c) = create_nonlinear_SVC('/Users/ezra/Documents/data_repo/creditcard.csv', 200,200)
    #print('------------------------------------------------------------------')
    #print('Optimal SVC -> Kernel {} | degree {} | C {}'.format(kernel, deg, c))
    return (kernel, deg, c)

if __name__ == "__main__":
    main()
