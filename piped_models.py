import os
import random
import numpy as np

from time import time
from sklearn.linear_model import RidgeClassifier

from clean_test import prepare_training_set, vectorise_dataset, \
    train_all_models, plot_results

from feature_engeneering import tile_features


def get_ridge_features(X_train, y_train, to_predict):
    clf, _ = RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"
    clf.fit(X_train, y_train)
    pred = clf.predict(to_predict)
    return (pred - 1) * 0.01


def make_features(X_train, X_test, y_train, language):
    X_train_v, X_test_v = vectorise_dataset(X_train, X_test, y_train)

    ridge_features = get_ridge_features(X_train_v, y_train, X_train_v)
    custom_features = np.array([tile_features(w, language) for w in X_train])
    X_train = np.column_stack((custom_features, ridge_features))

    # TODO eliminate copy/paste
    ridge_features = get_ridge_features(X_train_v, y_train, X_test_v)
    custom_features = np.array([tile_features(w, language) for w in X_test])
    X_test = np.column_stack((custom_features, ridge_features))

    return X_train, X_test


def main():
    random.seed(42)

    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko"
    language = "RUS"
    # aut_pair = "asimov-vs-silverberg"

    train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    X_train, y_train = prepare_training_set(train_fn1, train_fn2)

    test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")

    X_test, y_test = prepare_training_set(test_fn1, test_fn2)

    n = len(X_train) // 100
    mini_X_train, mini_X_test, mini_y_train, mini_y_test = (x[:n] for x in (X_train, X_test, y_train, y_test))
    mini_X_train_v, _ = vectorise_dataset(mini_X_train, mini_X_test, mini_y_train)

    t = time()
    mini_X_train_f, mini_X_test_f = make_features(mini_X_train, mini_X_test, mini_y_train, language)
    results = train_all_models(mini_X_train_f, mini_y_train, mini_X_test_f, mini_y_test)
    plot_results(results)
    print("expected time {}".format((time() - t) * 100 // 60))

    # X_train, X_test = make_features(X_train, X_test, y_train, language)
    # results = train_all_models(X_train, y_train, X_test, y_test)
    # plot_results(results)

    # time 10.0
    # print("time {}".format((time() - t) // 60))

if __name__ == '__main__':
    main()
