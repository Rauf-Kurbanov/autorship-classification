import pickle
from time import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid

from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.svm import LinearSVC

import xgboost as xgb
from sklearn.utils.extmath import density

import matplotlib.pyplot as plt

from feature_engineering import BagOfWords
from prep import deseralize_dataset
from prep import seralize_dataset


def proportion(pred):
    na, nb = np.histogram(pred, bins=2)[0]
    return na / (na + nb), nb / (na + nb)


def predict_and_score(X, y, data_name, clf):
    t0 = time()
    pred = clf.predict(X)
    test_time = time() - t0
    print(data_name + " time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y, pred)
    print("accuracy on " + data_name + ":   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report for " + data_name + ":")
    print(metrics.classification_report(y, pred))

    print("Classes proportion on " + data_name + ": {:1.2f} / {:1.2f}".format(*proportion(pred)))
    print()

    return score


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    train_score = predict_and_score(X_train, y_train, "train", clf)
    test_score = predict_and_score(X_test, y_test, "test", clf)

    return train_score, test_score


def train_models(X_train, y_train, X_test, y_test, models_with_names):
    results = []

    for clf, name in models_with_names:

        print('=' * 80)
        print(name)
        results.append((name,) + benchmark(clf, X_train, y_train, X_test, y_test))

    return results


def train_all_models(X_train, y_train, X_test, y_test):
    piped = Pipeline([
        ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3))
        , ('classification', LinearSVC())])

    models_with_names = (
            (RidgeClassifier(tol=1e-2, solver="sag", alpha=10000), "Ridge Classifier")
            , (Perceptron(n_iter=50, penalty='l2', alpha=0.001), "Perceptron")
            , (PassiveAggressiveClassifier(n_iter=50, C=1e-8), "Passive-Aggressive")
            , (SGDClassifier(alpha=.002, n_iter=50, penalty="elasticnet"), "LinearSVC with elastic-Net penalty")
            , (NearestCentroid(), "NearestCentroid (aka Rocchio classifier)")
            , (MultinomialNB(alpha=.01), "MultinomialNB")
            , (BernoulliNB(alpha=.01), "BernoulliNB")
            , ((xgb.XGBClassifier(max_depth=5, n_estimators=50, learning_rate=0.02)), "XGBoost")
            , (RandomForestClassifier(n_estimators=50, max_depth=3, max_features=20), "Random forest")
            , (piped, "LinearSVC with L1-based feature selection")
            , (LinearSVC(penalty="l2", dual=False, tol=1e-3), "LinearSVC with l2 penalty")
            , (SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"), "SGDClassifier")
            , (LinearSVC(), "LinearSVC"))

    return train_models(X_train, y_train, X_test, y_test, models_with_names)


def plot_results(results):
    indices = np.arange(len(results))
    results = [[x[i] for x in results] for i in range(3)]

    clf_names, train_score, test_score = results

    plt.figure(figsize=(20, 8))
    plt.title("Score")
    plt.barh(indices, train_score, .2, label="training score", color='r')
    plt.barh(indices + .3, test_score, .2, label="test score", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


def main():
    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko_mini"
    nfeatures = 1000
    words_per_tile = 100
    vectorised = True
    force = False
    shrink_train = 10
    shrink_test = 10

    suffix = "{}_nf={}_wpt={}_vec={}_shTr={}_shTst={}".format(aut_pair, nfeatures, words_per_tile, vectorised,
                                                              shrink_train, shrink_test)
    seralize_dataset(data_dir, aut_pair, nfeatures, suffix=suffix, words_per_tile=words_per_tile,
                     vectoriser=BagOfWords(), force=force, shrink_train=shrink_train, shrink_test=shrink_test)
    X_train, y_train, X_test, y_test = deseralize_dataset(suffix)

    dlen = X_train.shape[0]
    X_train, y_train = X_train[:dlen, :], y_train[:dlen]
    print("Cropped X_train.shape = {}".format(X_train.shape))

    results = train_all_models(X_train, y_train, X_test, y_test)
    pickle.dump(results, open("serialized/results" + suffix + ".p", "wb"))

    # results = pickle.load(open("serialized/results" + suffix + ".p", "rb"))
    plot_results(results)

if __name__ == '__main__':
    main()
