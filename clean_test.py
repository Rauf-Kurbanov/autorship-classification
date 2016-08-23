import io
from time import time

from sklearn.ensemble import RandomForestClassifier
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
import itertools as itt
import pickle

import xgboost as xgb


def proportion(pred):
    na, nb = np.histogram(pred, bins=2)[0]
    return na / (na + nb), nb / (na + nb)


def build_tiles(text, nwords=10):
    ll = [s.strip().split() for s in text.splitlines()]
    words = list(itt.chain(*ll))
    word_lists = [words[i:i+nwords] for i, _ in enumerate(words[:-(nwords-1)])]
    return [" ".join(wl) for wl in word_lists]


def prepare_training_set(filename1, filename2):
    print("Opening files:\n%s\n%s" % (filename1, filename2))
    with io.open(filename1) as f1, \
            io.open(filename2) as f2:
        text1 = f1.read()
        text2 = f2.read()

    print("Size of the first text = %d" % len(text1))
    print("Size of the second text = %d" % len(text2))

    t0 = time()
    tiles1 = build_tiles(text1)
    print("broke text1 in tiles in %fs" % (time() - t0))

    t1 = time()
    tiles2 = build_tiles(text2)
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
    labels1 = np.array([1 for _ in range(len(tiles1))])
    labels2 = np.array([2 for _ in range(len(tiles2))])
    labels = np.append(labels1, labels2)

    index_shuf = list(range(len(tiles)))
    shuffle(index_shuf)
    tiles = [tiles[i] for i in index_shuf]
    labels = [labels[i] for i in index_shuf]

    return tiles, labels


def vectorise_dataset(X_train, x_test, y_train, select_chi2=1000):
    n_features = 2 ** 16
    # n_features = 1000

    hasher = HashingVectorizer(stop_words='english',
                               non_negative=True,
                               n_features=n_features)

    vectoriser = make_pipeline(hasher, TfidfTransformer())
    # vectoriser = TfidfTransformer()

    X_train = vectoriser.fit_transform(X_train)
    x_test = vectoriser.transform(x_test)

    print("Extracting %d best features by a chi-squared test" % select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    x_test = ch2.transform(x_test)
    print("done in %fs \n" % (time() - t0))

    return X_train, x_test


# Benchmark classifiers
def benchmark(clf, X_train, y_train, X_test, y_test):
    res = []

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_train)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_train, pred)
    print("accuracy on train:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report for train:")
    print(metrics.classification_report(y_train, pred))

    print("Classes proportion on train: {}".format(proportion(pred)))

    print()



    # clf_descr = "TRAIN " + str(clf).split('(')[0]
    train_score = score
    # res.append((clf_descr, score, train_time, test_time))

    # TODO eliminate code duplicate
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy on test:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report for test:")
    print(metrics.classification_report(y_test, pred))
    print("Classes proportion on test: {}".format(proportion(pred)))

    print()


    clf_descr = str(clf).split('(')[0]
    test_score = score
    # res.append((clf_descr, score, train_time, test_time))

    # return clf_descr, score, train_time, test_time
    return clf_descr, train_score, test_score
    # return res


def train_all_models(X_train, y_train, X_test, y_test):
    results = []

    piped = Pipeline([
        ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3))
        , ('classification', LinearSVC())])
    for clf, name in [(RandomForestClassifier(n_estimators=300, max_depth=5, max_features=15), "Random forest")]:
    # for clf, name in (
    #         (RidgeClassifier(tol=1e-2, solver="sag", alpha=1000), "Ridge Classifier")
    #         , (Perceptron(n_iter=50, penalty='l2', alpha=0.001), "Perceptron")
            # , (PassiveAggressiveClassifier(n_iter=50, C=0.0001), "Passive-Aggressive")
            # # , (KNeighborsClassifier(n_neighbors=10), "kNN")
            # , (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), "Elastic-Net penalty")
            # , (NearestCentroid(), "NearestCentroid (aka Rocchio classifier)")
            # , (MultinomialNB(alpha=.01), "MultinomialNB")
            # , (BernoulliNB(alpha=.01), "BernoulliNB")
            # , ((xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)), "XGBoost")
            # , (RandomForestClassifier(n_estimators=300, max_depth=3, max_features=15), "Random forest")
            # , (piped, "LinearSVC with L1-based feature selection")
    # ):

        print('=' * 80)
        print(name)
        results.append(benchmark(clf, X_train, y_train, X_test, y_test))
        # results.extend(benchmark(clf, X_train, y_train, X_test, y_test))

    ## for penalty in ["l2", "l1"]:
    # for penalty in ["l2"]:
    #     for clf in (LinearSVC(penalty=penalty, dual=False, tol=1e-3)
    #                 , SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)):
    #         print('=' * 80)
    #         print("%s penalty" % penalty.upper())
    #         results.append(benchmark(clf, X_train, y_train, X_test, y_test))
    #         # results.extend(benchmark(clf, X_train, y_train, X_test, y_test))

    return results


def plot_results(results):
    # %matplotlib inline
    indices = np.arange(len(results))

    # results = [[x[i] for x in results] for i in range(4)]
    results = [[x[i] for x in results] for i in range(3)]

    # clf_names, score, training_time, test_time = results
    # clf_names, score, training_time, _ = results
    clf_names, train_score, test_score = results

    # training_time = np.array(training_time) / np.max(training_time)
    # test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(20, 8))
    plt.title("Score")
    plt.barh(indices, train_score, .2, label="training score", color='r')
    plt.barh(indices + .3, test_score, .2, label="test score", color='b')
    # plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    # plt.subplots_adjust()
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


def main():
    # random.seed(42)

    # data_dir = "/home/rauf/Programs/shpilman/data"
    # aut_pair = "perumov-vs-lukjarenko"
    # # aut_pair = "asimov-vs-silverberg"
    #
    # train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    # train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    # X_train, y_train = prepare_training_set(train_fn1, train_fn2)
    #
    # test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    # test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")
    #
    # select_chi2 = 10000
    # X_test, y_test = prepare_training_set(test_fn1, test_fn2)
    # X_train, X_test = vectorise_dataset(X_train, X_test, y_train, select_chi2)
    #
    # pickle.dump(X_train, open("serialized/X_train.p", "wb"))
    # pickle.dump(y_train, open("serialized/y_train.p", "wb"))
    # pickle.dump(X_test, open("serialized/X_test.p", "wb"))
    # pickle.dump(y_test, open("serialized/y_test.p", "wb"))

    if all(map(os.path.isfile, ["serialized/X_train.p", "serialized/y_train.p",
                            "serialized/X_test.p", "serialized/y_test.p"])):
        X_train = pickle.load(open("serialized/X_train.p", "rb"))
        y_train = pickle.load(open("serialized/y_train.p", "rb"))
        X_test = pickle.load(open("serialized/X_test.p", "rb"))
        y_test = pickle.load(open("serialized/y_test.p", "rb"))
        print("X_train.shape = {}".format(X_train.shape))
    else:
        print("Serialized files not found")
        return

    dlen = X_train.shape[0] // 10
    # X_train, y_train = X_train[:dlen, :], y_train[:dlen]

    results = train_all_models(X_train, y_train, X_test, y_test)
    plot_results(results)

if __name__ == "__main__":
    main()
