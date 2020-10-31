import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
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
n_samples = X.shape[0]

pca = PCA()
X_transformed = pca.fit_transform(X)
X_r = pca.fit_transform(X)

X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
    print(eigenvalue)

#lda = LinearDiscriminantAnalysis(n_components=2)
#X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
print(pca.get_covariance())
plt.figure()
colors = ['navy', 'turquoise']
lw = 2

for color, i, target_name in zip(colors, [0, 1], ['no', 'yes']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Second Dataset (Bank Data)')
plt.xlabel("feature 1 (transformed)")
plt.ylabel("feature 2 (transformed")

plt.clf()

loss_arr = []
fig, (ax1) = plt.subplots(1, 1, sharex=True)
for i in range(X.shape[1]):
    print("Doing {} components...".format(i+1))
    pca = PCA(n_components=i)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    loss_arr.append(((X - X_reconstructed) ** 2).mean().sum())
ax1.plot(np.arange(X.shape[1]), loss_arr, label="reconstruction error")
ax1.set_ylabel("Avg RMSE")
plt.xlabel("n_components")
fig.suptitle("Reconstruction Error vs PCA Dimensionality Reduction DS1")
plt.savefig("pcaerr1.png")