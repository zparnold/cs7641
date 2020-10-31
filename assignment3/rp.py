import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, kurtosis
import seaborn as sns
from sklearn.random_projection import GaussianRandomProjection

def run_thingy(X, name):
    avg_loss_arr = []
    for x in np.arange(10):
        loss_arr = []
        for i in range(X.shape[1]):
            ica = GaussianRandomProjection(n_components=i + 1)
            X_transformed = ica.fit_transform(X)
            inverse_data = np.linalg.pinv(ica.components_.T)
            reconstructed_data = X_transformed.dot(inverse_data)
            loss_arr.append(((X - reconstructed_data) ** 2).mean().sum())
        if x == 10:
            fig, (ax1) = plt.subplots(1, 1, sharex=True)
            ax1.plot(np.arange(X.shape[1]), loss_arr, label="reconstruction error")
            ax1.set_ylabel("Avg RMSE")
            plt.xlabel("n_components")
            fig.suptitle("Reconstruction Error RP")
            plt.savefig(name)
        avg_loss_arr.append(np.average(loss_arr))
    print("Variance: {} Std: {}".format(np.var(avg_loss_arr), np.std(avg_loss_arr)))

def main():
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                               'native-country', '<=50k']
    df = pd.read_csv('./adult-small.data',
                      names=features)
    df.dropna()
    df.drop_duplicates()
    df = df[df['workclass'] != '?']
    df = df[df['occupation'] != '?']
    df = df[df['education'] != '?']
    df = df[df['marital-status'] != '?']
    df = df[df['relationship'] != '?']
    df = df[df['race'] != '?']
    df = df[df['sex'] != '?']
    df = df[df['native-country'] != '?']
    X = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    X['<=50k'] = X['<=50k'].map({'<=50K':1, '>50K': 0})
    y = X['<=50k']
    X = X.drop(['<=50k'], axis=1)
    run_thingy(X, "rpds1.png")

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
    run_thingy(X, "rpds2.png")


if __name__ == '__main__':
    main()