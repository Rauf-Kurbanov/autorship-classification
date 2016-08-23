import io
import pickle
import os
import random
import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
from time import time
from random import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.pipeline import make_pipeline

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

    hasher = HashingVectorizer(stop_words='english',
                               non_negative=True,
                               n_features=n_features)

    vectoriser = make_pipeline(hasher, TfidfTransformer())

    X_train = vectoriser.fit_transform(X_train)
    x_test = vectoriser.transform(x_test)

    print("Extracting %d best features by a chi-squared test" % select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    x_test = ch2.transform(x_test)
    print("done in %fs \n" % (time() - t0))

    return X_train, x_test


def benchmark(clf, X_train, y_train, X_test, y_test):

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    def predict_and_score(X, y, data_name):
        t0 = time()
        pred = clf.predict(X)
        test_time = time() - t0
        print(data_name + " time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y, pred)
        print("accuracy on" + data_name + ":   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

        print("classification report for " + data_name + ":")
        print(metrics.classification_report(y, pred))
        print("Classes proportion on " + data_name + ": {:1.2f} / {:1.2f}".format(*proportion(pred)))
        print()

        return score

    train_score = predict_and_score(X_train, y_train, "train")
    test_score = predict_and_score(X_test, y_test, "test")

    clf_descr = str(clf).split('(')[0]

    return clf_descr, train_score, test_score


def train_all_models(X_train, y_train, X_test, y_test):
    results = []

    piped = Pipeline([
        ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3))
        , ('classification', LinearSVC())])

    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="sag", alpha=1000), "Ridge Classifier")
            , (Perceptron(n_iter=50, penalty='l2', alpha=0.001), "Perceptron")
            , (PassiveAggressiveClassifier(n_iter=50, C=0.0001), "Passive-Aggressive")
            , (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), "Elastic-Net penalty")
            , (NearestCentroid(), "NearestCentroid (aka Rocchio classifier)")
            , (MultinomialNB(alpha=.01), "MultinomialNB")
            , (BernoulliNB(alpha=.01), "BernoulliNB")
            , ((xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)), "XGBoost")
            , (RandomForestClassifier(n_estimators=300, max_depth=5, max_features=15), "Random forest")
            , (piped, "LinearSVC with L1-based feature selection")
    ):

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
    plt.savefig('foo.png')


def seralize_dataset(data_dir, aut_pair, nfeatures, suffix=""):
    if all(map(os.path.isfile, ["serialized/X_train" + suffix + ".p",
                                "serialized/y_train" + suffix + ".p",
                                "serialized/X_test" + suffix + ".p",
                                "serialized/y_test" + suffix + ".p"])):
        print("Serialized files already have been found")
        return

    train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    X_train, y_train = prepare_training_set(train_fn1, train_fn2)

    test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")

    X_test, y_test = prepare_training_set(test_fn1, test_fn2)
    X_train, X_test = vectorise_dataset(X_train, X_test, y_train, nfeatures)

    pickle.dump(X_train, open("serialized/X_train" + suffix + ".p", "wb"))
    pickle.dump(y_train, open("serialized/y_train" + suffix + ".p", "wb"))
    pickle.dump(X_test, open("serialized/X_test" + suffix + ".p", "wb"))
    pickle.dump(y_test, open("serialized/y_test" + suffix + ".p", "wb"))


def deseralize_dataset(suffix):
    print("Serializing dataset " + suffix)
    t = time()
    with open("serialized/X_train" + suffix + ".p", 'rb') as X_train_f,\
            open("serialized/y_train" + suffix + ".p", 'rb') as y_train_f,\
            open("serialized/X_test" + suffix + ".p", 'rb') as X_test_f,\
            open("serialized/y_test" + suffix + ".p", 'rb') as y_test_f:
        X_train = pickle.load(X_train_f)
        y_train = pickle.load(y_train_f)
        X_test = pickle.load(X_test_f)
        y_test = pickle.load(y_test_f)
    print("Serialization finished in {}s".format(time() - t))

    print("X_train.shape = {}".format(X_train.shape))
    return X_train, y_train, X_test, y_test


def main():
    # random.seed(42)
    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko"
    # aut_pair = "asimov-vs-silverberg"
    nfeatures = 10000
    suffix = str(nfeatures) + aut_pair

    seralize_dataset(data_dir, aut_pair, nfeatures, suffix)
    X_train, y_train, X_test, y_test = deseralize_dataset(suffix)

    dlen = X_train.shape[0] // 100
    X_train, y_train = X_train[:dlen, :], y_train[:dlen]
    print("Cropped X_train.shape = {}".format(X_train.shape))

    results = train_all_models(X_train, y_train, X_test, y_test)
    pickle.dump(results, open("serialized/results" + suffix + ".p", "wb"))

    plot_results(results)

if __name__ == "__main__":
    main()
