import sys

import six

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def run_something(values, restarts=0, schedule=mlrose.GeomDecay(), mutation_prob=0.1, pop_size=200):
    curves = []
    for x in ['identity', 'relu', 'sigmoid', 'tanh']:
        # Normalize feature data
        if values['algo'] == 'rhc':
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[10, 10], activation=x,
                                             algorithm='random_hill_climb', max_iters=100,
                                             bias=True, is_classifier=True, learning_rate=0.00001,
                                             early_stopping=True, max_attempts=100,
                                             random_state=1, restarts=restarts, curve=True, clip_max=1)
        if values['algo'] == 'sa':
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[10, 10], activation=x,
                                             algorithm='simulated_annealing', max_iters=100,
                                             bias=True, is_classifier=True, learning_rate=0.00001,
                                             early_stopping=True, max_attempts=100,
                                             random_state=1, schedule=schedule, curve=True, clip_max=1)
        if values['algo'] == 'ga':
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[10, 10], activation=x,
                                             algorithm='genetic_alg', max_iters=100,
                                             bias=True, is_classifier=True, learning_rate=0.00001,
                                             early_stopping=True, max_attempts=100,
                                             random_state=1, pop_size=pop_size, mutation_prob=mutation_prob, curve=True,
                                             clip_max=1)
        print('begin training')
        nn_model1.fit(values["X_train"], values["y_train"])
        # Predict labels for train set and assess accuracy
        y_train_pred = nn_model1.predict(values["X_train"])

        y_train_accuracy = accuracy_score(values["y_train"], y_train_pred)

        print('Training accuracy: ', y_train_accuracy)

        # Predict labels for test set and assess accuracy
        y_test_pred = nn_model1.predict(values["X_test"])

        y_test_accuracy = accuracy_score(values["y_test"], y_test_pred)

        print('Test accuracy: ', y_test_accuracy)

        print('Loss function value', nn_model1.loss)
        curves.append(nn_model1.fitness_curve)
    return curves


def main():
    # ds1 = pd.read_csv('../assignment1/data/adult.data',
    #                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    #                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
    #                          'native-country', '<=50k'])
    # ds1.dropna()
    # ds1.drop_duplicates()
    # ds1 = ds1[ds1['workclass'] != '?']
    # ds1 = ds1[ds1['occupation'] != '?']
    # ds1 = ds1[ds1['education'] != '?']
    # ds1 = ds1[ds1['marital-status'] != '?']
    # ds1 = ds1[ds1['relationship'] != '?']
    # ds1 = ds1[ds1['race'] != '?']
    # ds1 = ds1[ds1['sex'] != '?']
    # ds1 = ds1[ds1['native-country'] != '?']
    # ds1_dummies = pd.get_dummies(ds1, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship',
    #                                            'race', 'sex', 'native-country'])
    # ds1_dummies.dropna()
    # ds1_dummies['<=50k'].value_counts()
    # ds1_dummies['<=50k'] = ds1_dummies['<=50k'].map({'<=50K': 1, '>50K': 0})
    # ds1_labels = ds1_dummies['<=50k']
    # ds1_dummies = ds1_dummies.drop(['<=50k'], axis=1)
    ds2 = pd.read_csv('../assignment1/data/bank-additional-full.csv', delimiter=';')
    ds2.dropna()
    ds2.drop_duplicates()
    ds2_dummies = pd.get_dummies(ds2, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                                               'month', 'day_of_week', 'poutcome'])
    ds2_dummies.dropna()
    ds2_dummies['y'].value_counts()
    ds2_dummies['y'] = ds2_dummies['y'].map({'yes': 1, 'no': 0})
    ds2_labels = ds2_dummies['y']
    ds2_dummies = ds2_dummies.drop(['y'], axis=1)
    scaler = StandardScaler()

    scaled_ds1_dummies = scaler.fit_transform(ds2_dummies, y=ds2_labels)

    X_train, X_test, y_train, y_test = train_test_split(scaled_ds1_dummies, ds2_labels, test_size=0.20,
                                                        stratify=ds2_labels)

    v = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    print("Random Hill Climbing")
    v['algo'] = 'rhc'
    rhc = run_something(v, 0)
    print(rhc)
    print("SA")

    v['algo'] = 'sa'
    sa = run_something(v, 0, mlrose.ExpDecay(), 0.1)
    print(sa)
    print("GA")

    v['algo'] = 'ga'
    ga = run_something(v, 0, mlrose.ExpDecay(), 0.1, 100)

    f, axarr = plt.subplots(1, 3)
    f.set_figheight(3)
    f.set_figwidth(9)
    for y, i  in zip(rhc, ['r', 'b', 'g', 'y']):
        axarr[0].plot(y, color=i)
    axarr[0].set_title('RHC vs Activation Function')
    for y, i  in zip(sa, ['r', 'b', 'g', 'y']):
        axarr[1].plot(y, color=i)
    axarr[1].set_title('SA vs Activation Function')
    for y, i  in zip(ga, ['r', 'b', 'g', 'y']):
        axarr[2].plot(y, color=i)
    axarr[2].set_title('GA vs Activation Function')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.legend(handles=[Line2D([0], [0], color='r', lw=4, label='identity'),
                        Line2D([0], [0], color='b', lw=4, label='relu'),
                        Line2D([0], [0], color='g', lw=4, label='sigmoid'),
                        Line2D([0], [0], color='y', lw=4, label='tanh')])
    # plt.title('Input size vs fitness curve One Max')
    # plt.xlabel('Function iteration count')
    # plt.ylabel('Fitness function value')

    plt.show()


if __name__ == '__main__':
    main()
