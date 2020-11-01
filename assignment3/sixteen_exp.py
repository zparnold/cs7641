import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, kurtosis
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.model_selection as pre

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

