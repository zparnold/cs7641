import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, kurtosis
import seaborn as sns

def main():
    # features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    #                            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
    #                            'native-country', '<=50k']
    # df = pd.read_csv('./adult-small.data',
    #                   names=features)
    # df.dropna()
    # df.drop_duplicates()
    # df = df[df['workclass'] != '?']
    # df = df[df['occupation'] != '?']
    # df = df[df['education'] != '?']
    # df = df[df['marital-status'] != '?']
    # df = df[df['relationship'] != '?']
    # df = df[df['race'] != '?']
    # df = df[df['sex'] != '?']
    # df = df[df['native-country'] != '?']
    # X = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    # X['<=50k'] = X['<=50k'].map({'<=50K':1, '>50K': 0})
    # y = X['<=50k']
    # X = X.drop(['<=50k'], axis=1)

    df = pd.read_csv('./bank-additional.csv', delimiter=';')
    df.dropna()
    df.drop_duplicates()
    X = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                    'day_of_week', 'poutcome'])
    X.dropna()
    X['y'].value_counts()
    X['y'] = X['y'].map({'yes': 1, 'no': 0})
    y = X['y']
    X = X.drop(['y'], axis=1)

    kurt_arr = []
    loss_arr = []
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for i in range(X.shape[1]):
        print("Doing {} components...".format(i+1))
        ica = FastICA(whiten=True, n_components=i+1)
        X_transformed = ica.fit_transform(X)
        print(kurtosis(X_transformed, axis=1))
        print(np.average(kurtosis(X_transformed, axis=1)))
        kurt_arr.append(
            np.average(kurtosis(X_transformed, axis=1))
        )
        X_reconstructed = ica.inverse_transform(X_transformed)
        loss_arr.append(((X - X_reconstructed) ** 2).mean().sum())

    ax1.plot(np.arange(X.shape[1]), kurt_arr, label="kurtosis")
    ax1.set_ylabel("Avg Kurtosis")
    ax2.plot(np.arange(X.shape[1]), loss_arr, label="reconstruction error")
    ax2.set_ylabel("Avg RMSE")
    plt.xlabel("n_components")
    fig.suptitle("Kurtosis vs ICA Dimensionality Reduction DS2")
    plt.savefig("icads2.png")

if __name__ == '__main__':
    main()