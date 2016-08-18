import io
from time import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from random import shuffle
# from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
# from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density

from sklearn import metrics
from sklearn.pipeline import make_pipeline

import os
import random


def merge_by(s, num_to_merge=2):
    # num_to_merge = 2
    new_s = []
    for i in range(0, len(s) - len(s) % num_to_merge, num_to_merge):
        ss = s[i:i+num_to_merge]
        ss = ss[0] + ss[1]
        new_s.append(ss)
    return new_s


def prepare_training_set(filename1, filename2, num_to_merge=2):
    print ("Opening files:\n%s\n%s" % (filename1, filename2))
    with io.open(filename1) as f1, \
            io.open(filename2) as f2:
        text1 = f1.read()
        text2 = f2.read()

    print("Size of the first text = %d" % len(text1))
    print("Size of the second text = %d" % len(text2))

    t0 = time()
    tiles1 = text1.split("\n")
    tiles1 = [t for t in tiles1 if len(t) > 0]
    tiles1 = merge_by(tiles1, num_to_merge)

    print("broke text1 in tiles in %fs" % (time() - t0))

    t1 = time()
    tiles2 = text2.split("\n")
    tiles2 = [t for t in tiles2 if len(t) > 0]
    tiles2 = merge_by(tiles2, num_to_merge)

    print("broke text2 in tiles in"
          " %fs" % (time() - t1))

    print("Number of samples from first text = %d" % len(tiles1))
    print("Number of samples from second text = %d" % len(tiles2))

    # making equal
    min_samples = min(len(tiles1), len(tiles2))
    tiles1 = tiles1[:min_samples]
    tiles2 = tiles2[:min_samples]

    print("Cropped number of samples = %d" % len(tiles1))

    tiles = tiles1 + tiles2
    # for i in range(10):               ###
    #     ind = random.randint(0, len(tiles) - 1)
    #     print(tiles[ind], end="\n----------------------------------\n")

    labels1 = np.array([1 for _ in range(len(tiles1))])
    labels2 = np.array([2 for _ in range(len(tiles2))])
    labels = np.append(labels1, labels2)

    index_shuf = list(range(len(tiles)))
    shuffle(index_shuf)
    tiles = [tiles[i] for i in index_shuf]
    labels = [labels[i] for i in index_shuf]

    return tiles, labels


def vectorise_dataset(X_train, x_test, y_train, select_chi2=10000):
    n_features = 2 ** 16

    hasher = HashingVectorizer(stop_words='english',
                               non_negative=True,
                               n_features=n_features)

    vectorizer = make_pipeline(hasher, TfidfTransformer())

    X_train = vectorizer.fit_transform(X_train)
    x_test = vectorizer.transform(x_test)

    print("Extracting %d best features by a chi-squared test" % select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    x_test = ch2.transform(x_test)
    print("done in %fs \n" % (time() - t0))

    return X_train, x_test


# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred))

    print()

    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time


def train_all_models(X_train, y_train, X_test, y_test):
    results = []

    piped = Pipeline([
        ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3))
        , ('classification', LinearSVC())])
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier")
            , (Perceptron(n_iter=50), "Perceptron")
            , (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive")
            # , (KNeighborsClassifier(n_neighbors=10), "kNN")
            , (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), "Elastic-Net penalty")
            , (NearestCentroid(), "NearestCentroid (aka Rocchio classifier)")
            , (MultinomialNB(alpha=.01), "MultinomialNB")
            , (BernoulliNB(alpha=.01), "BernoulliNB")
            , (piped, "LinearSVC with L1-based feature selection")):
        #         (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, X_train, y_train, X_test, y_test))

    for penalty in ["l2", "l1"]:
        for clf in (LinearSVC(penalty=penalty, dual=False, tol=1e-3)
                    , SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)):
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            results.append(benchmark(clf, X_train, y_train, X_test, y_test))

    return results


def plot_results(results):
    # %matplotlib inline
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


def main():
    random.seed(42)

    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko"
    # aut_pair = "asimov-vs-silverberg"

    train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    X_train, y_train = prepare_training_set(train_fn1, train_fn2)

    test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")

    X_test, y_test = prepare_training_set(test_fn1, test_fn2)
    X_train, X_test = vectorise_dataset(X_train, X_test, y_train)
    # print(type(X_train))

    # print(X_train[0])
    results = train_all_models(X_train, y_train, X_test, y_test)
    plot_results(results)

if __name__ == "__main__":
    main()
