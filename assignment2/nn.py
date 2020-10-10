import sys

import six

sys.modules['sklearn.externals.six'] = six
import mlrose
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def main():
    ds1 = pd.read_csv('../assignment1/data/adult.data',
                      names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                             'native-country', '<=50k'])
    ds1.dropna()
    ds1.drop_duplicates()
    ds1 = ds1[ds1['workclass'] != '?']
    ds1 = ds1[ds1['occupation'] != '?']
    ds1 = ds1[ds1['education'] != '?']
    ds1 = ds1[ds1['marital-status'] != '?']
    ds1 = ds1[ds1['relationship'] != '?']
    ds1 = ds1[ds1['race'] != '?']
    ds1 = ds1[ds1['sex'] != '?']
    ds1 = ds1[ds1['native-country'] != '?']
    ds1_dummies = pd.get_dummies(ds1, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                               'race', 'sex', 'native-country'])
    ds1_dummies.dropna()
    ds1_dummies['<=50k'].value_counts()
    ds1_dummies['<=50k'] = ds1_dummies['<=50k'].map({'<=50K': 1, '>50K': 0})
    ds1_labels = ds1_dummies['<=50k']
    ds1_dummies = ds1_dummies.drop(['<=50k'], axis=1)
    scaler = StandardScaler()

    scaled_ds1_dummies = scaler.fit_transform(ds1_dummies, y=ds1_labels)

    X_train, X_test, y_train, y_test = train_test_split(scaled_ds1_dummies, ds1_labels, test_size=0.20,
                                                        stratify=ds1_labels)
    # Normalize feature data
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[10], activation='relu',
                                     algorithm='random_hill_climb', max_iters=1000,
                                     bias=True, is_classifier=True, learning_rate=0.0001,
                                     early_stopping=True, clip_max=5, max_attempts=1000,
                                     random_state=1)

    print('begin training')
    nn_model1.fit(X_train, y_train)
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train)

    y_train_accuracy = accuracy_score(y_train, y_train_pred)

    print('Training accuracy: ', y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test)

    y_test_accuracy = accuracy_score(y_test, y_test_pred)

    print('Test accuracy: ', y_test_accuracy)


if __name__ == '__main__':
    main()
