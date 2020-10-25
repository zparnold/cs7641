import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import sklearn.metrics as metrics
# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
# Only take the first fold.
df = pd.read_csv('./bank-additional.csv', delimiter=';')
df.dropna()
df.drop_duplicates()
X = pd.get_dummies(df, columns=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'])
X.dropna()
X['y'].value_counts()
X['y'] = X['y'].map({'yes':1, 'no': 0})
y = X['y']
X = X.drop(['y'], axis=1)

X_train, y_train, X_test, y_test = train_test_split(X, y, stratify=y, random_state=0)

n_components = np.arange(1, 2)
models = [GMM(n, covariance_type='full', random_state=0).fit(X_train)
          for n in n_components]

plt.plot(n_components, [m.bic(X_train) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_train) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
model = GMM(3, covariance_type='full', random_state=0).fit(X)
cluster_labels = model.predict(X)
print('NMI: {}'.format(metrics.normalized_mutual_info_score(y, cluster_labels)))
print('Homogeneity: {}'.format(metrics.homogeneity_score(y, cluster_labels)))
print('Completeness: {}'.format(metrics.completeness_score(y, cluster_labels)))
