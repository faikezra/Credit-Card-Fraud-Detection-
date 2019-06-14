import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


import neural_network
import roc_split
import log_reg_selection
import svm


def cm(y_true, y_pred,classes, normalize, title):

    confusion = confusion_matrix(y_true, y_pred)
    if normalize:
        confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

kernel, deg, c = svm.main()
roc_threshold = roc_split.main()
connections, hidden_layers = neural_network.main()
C, regulizer = log_reg_selection.main()
print("----------------------------------------")
print('Optimal SVC -> Kernel {} | degree {} | C {}'.format(kernel, deg, c))
print('optimal ROC -> ROC threshold {}'.format(roc_threshold))
print('optimal MLP -> Neural Connections {} | Hidden Layers {}'.format(connections, hidden_layers))
print('optimal Log -> C value {} | Regularizer {}'.format(C, regulizer))
print("----------------------------------------")

undersample, full_test = preprocessing.create_datasets('/Users/ezra/Documents/data_repo/creditcard.csv', 250,250)
train_x, validation_x, train_y, validation_y = undersample
test_x, test_y = full_test
# for the evaluation we will not use a validation set
train_x = pd.concat([train_x,validation_x])
train_y = pd.concat([train_y,validation_y])

# models
log_reg = LogisticRegression(C = C, penalty=regulizer)
log_reg.fit(train_x, train_y.values.ravel())
SVM = SVC(C = c, degree=deg, kernel=kernel)
SVM.fit(train_x, train_y.values.ravel())
MLP = Sequential()
MLP.add(Dense(connections, input_dim = 30, activation='relu'))
for i in range(hidden_layers):
    MLP.add(Dense(connections, activation='relu'))
MLP.add(Dense(1, activation='softmax'))
MLP.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
MLP.fit(train_x, train_y.values.ravel(), epochs=200, batch_size=10,verbose=0)
# model predictions
y_true = test_y.values.ravel()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(test_x)
principalDf = pd.DataFrame(data = principalComponents
         , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, pd.DataFrame.reset_index(test_y).drop('index', axis = 1)], axis = 1)
normalizer = MinMaxScaler()
finalDf = pd.DataFrame(data = normalizer.fit_transform(finalDf), columns = ['principal component 1', 'principal component 2', 'Class'])
ROC_predicted = [1.0 if i > roc_threshold else 0.0 for i in list(finalDf['principal component 1'])]
SVM_predicted = SVM.predict(test_x)
MLP_predicted = MLP.predict(test_x)
LOG_predicted = log_reg.predict(test_x)
# Isolation Forest Without Undersampling
ISO = IsolationForest()
ISO.fit(test_x)
ISO_predicted = [1 if i==-1 else 0 for i in ISO.predict(test_x)]

# Recall Evaluation
print('ROC Recall Value: {}'.format(recall_score(y_true,ROC_predicted)))
print('SVM Recall Value: {}'.format(recall_score(y_true,SVM_predicted)))
print('MLP Recall Value: {}'.format(recall_score(y_true,MLP_predicted)))
print('LOG Recall Value: {}'.format(recall_score(y_true,LOG_predicted)))
print('Isolation Forest Recall Value: {}'.format(recall_score(y_true,ISO_predicted)))


# Precision Evaluation
print('ROC Precision Value: {}'.format(precision_score(y_true,ROC_predicted)))
print('SVM Precision Value: {}'.format(precision_score(y_true,SVM_predicted)))
print('MLP Precision Value: {}'.format(precision_score(y_true,MLP_predicted)))
print('LOG Precision Value: {}'.format(precision_score(y_true,LOG_predicted)))
print('Isolation Forest Precision Value: {}'.format(precision_score(y_true,ISO_predicted)))

# CM
np.set_printoptions(precision=2)
predictions = [ROC_predicted,SVM_predicted,MLP_predicted,LOG_predicted,ISO_predicted]
model_names = ['PCA_ROC Split', 'Support Vector Machine', 'Multi-Layer Perceptron', 'Logistic Regression',
               'Isolation Forest']
for prediction, model_name in zip(predictions,model_names):
    cm(y_true, prediction, classes=[0,1], normalize=True, title='{} Confusion Matrix'.format(model_name))
    plt.show()
