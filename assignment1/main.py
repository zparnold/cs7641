import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


def main():

    X = pd.read_csv('./data/adult.data', names=['age', 'workclass', 'fnlwgt','education', 'education-num', 'marital-status', 'occupation', 'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','<=50k'])
    X.dropna()
    X.drop_duplicates()
    X = X[X['workclass'] != '?']
    X = X[X['occupation'] != '?']
    X = X[X['education'] != '?']
    X = X[X['marital-status'] != '?']
    X = X[X['relationship'] != '?']
    X = X[X['race'] != '?']
    X = X[X['sex'] != '?']
    X = X[X['native-country'] != '?']
    le = LabelEncoder()
    X['<=50k'] = le.fit_transform(X['<=50k'])
    y = X['<=50k']
    X = X.drop(['<=50k'], axis=1)
    clf = tree.DecisionTreeClassifier(random_state=0)
    enc = OrdinalEncoder()
    X_trans = enc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()
if __name__ == '__main__':
    main()