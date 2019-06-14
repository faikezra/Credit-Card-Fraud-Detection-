import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statistics import mean
from sklearn import metrics
import pandas as pd

def plot_PCA (train_x, train_y, pca, plot):
    principalComponents = pca.fit_transform(train_x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, pd.DataFrame.reset_index(train_y).drop('index', axis = 1)], axis = 1)
    normalizer = MinMaxScaler()
    finalDf = pd.DataFrame(data = normalizer.fit_transform(finalDf), columns = ['principal component 1', 'principal component 2', 'Class'])
    # plotting below
    if plot:
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = [1, 0]
        colors = ['r', 'g']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['Class'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()
        plt.show()
    return finalDf

def PCA_(undersample_amount, data_dir, plot):
    train_x, test_x, train_y, test_y = preprocessing.create_datasets(data_dir, undersample_amount, undersample_amount)[0]
    pca = PCA(n_components=2)
    finalDf = plot_PCA(train_x, train_y, pca, plot)
    pca = PCA(n_components= 10)
    principalComponents = pca.fit_transform(train_x)
    variance_explained = pca.explained_variance_ratio_
    #print('First 10 Principal Components Variance Explained As Follows Respectively: {}'.format(variance_explained))

    # first two principal components consistantly (every stratified split sample) explains about 75% and above of the variance
    # while the first component explains about 65% of the whole thing
    return (finalDf, test_x, test_y)

def ROC(df,threshold_value, plot):
    fpr, tpr, thresholds = metrics.roc_curve(df['Class'], df['principal component 1'])
    roc_auc = metrics.roc_auc_score(df['Class'], df['principal component 1'])
    if plot:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
    # take split point that catches about 95% 0f the Frauds and makes false calls 30% of the time
    #print('True Positive Rate (Sensitivity) for train set: {} || False Positive Rate for train set: {}'.format(tpr[-17], fpr[-17]))
    return (thresholds[threshold_value])

def test_roc_split(split_point, test_x, test_y, plot):
    pca = PCA(n_components=2)
    finalDf = plot_PCA(test_x, test_y, pca, plot)
    pca = PCA(n_components= 10)
    principalComponents = pca.fit_transform(test_x)
    variance_explained = pca.explained_variance_ratio_
    #print('Test Set: First 10 Principal Components Variance Explained As Follows Respectively: {}'.format(variance_explained))
    predicted = [1.0 if i > split_point else 0.0 for i in list(finalDf['principal component 1'])]
    accuracy = accuracy_score(finalDf['Class'], predicted)
    #print('Test Set Performance: {}'.format(accuracy))
    return accuracy

def main():
    threshold_values_index = [-5, -10, -20, -30, -40]
    undersample_amount = 200
    data_dir = '/Users/ezra/Documents/data_repo/creditcard.csv'
    acc = []
    split = []
    for row in range(10):
        df, test_x, test_y = PCA_(undersample_amount,data_dir, False)
        keep_acc = []
        keep_split = []
        for threshold_values in threshold_values_index:
            first_principal_split_point = ROC(df, threshold_values, False)
            keep_acc.append(test_roc_split(first_principal_split_point, test_x, test_y, False))
            keep_split.append(first_principal_split_point)
        acc.append(mean(keep_acc))
        split.append(mean(keep_split))
    threshold = split[acc.index(max(acc))]
    #print("Optimal split point is: {}".format(threshold))
    pca = PCA(n_components=2)
    plot_PCA(test_x, test_y, pca, True)
    _ = ROC(df, 0, True)
    return threshold

if __name__ == "__main__":
    main()
