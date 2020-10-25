from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer

# Generate synthetic dataset with 8 random clusters
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
# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,20), metric='silhouette')

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

df = pd.read_csv('./bank-additional.csv', delimiter=';')
df.dropna()
df.drop_duplicates()
X = pd.get_dummies(df, columns=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'])
X.dropna()
X['y'].value_counts()
X['y'] = X['y'].map({'yes':1, 'no': 0})
y = X['y']
X = X.drop(['y'], axis=1)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,20), metric='silhouette')

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure