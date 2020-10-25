import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

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
ds1_dummies = pd.get_dummies(df, columns=['workclass','education','marital-status','occupation','relationship','race','sex','native-country'])
ds1_dummies['<=50k'] = ds1_dummies['<=50k'].map({'<=50K':1, '>50K': 0})

n_components = 4

pca = PCA(n_components=n_components)
components = pca.fit_transform(ds1_dummies)

total_var = pca.explained_variance_ratio_.sum() * 100

labels = {str(i): f"PC {i+1}" for i in range(n_components)}

fig = px.scatter_matrix(
    components,
    color=ds1_dummies["<=50k"],
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

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

target_names = ["<=50k", ">50k"]

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

#lda = LinearDiscriminantAnalysis(n_components=2)
#X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise']
lw = 2

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

#plt.figure()
#for color, i, target_name in zip(colors, [0, 1], target_names):
#    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#                label=target_name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('LDA of IRIS dataset')

plt.show()