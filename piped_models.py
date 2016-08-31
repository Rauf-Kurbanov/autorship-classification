from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier

from feature_set import FeatureSet, Encoder
from feature_engineering import tile_punct_features, LettersFeature, BagOfWords, word_length
from prep import seralize_dataset, deseralize_dataset
from test import plot_results, train_models


def main():
    language = "RUS"

    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko_mini"
    nfeatures = 1000
    words_per_tile = 100
    force = True
    shrink_train = 10
    shrink_test = 10

    fs = FeatureSet([tile_punct_features, LettersFeature(language), word_length])
    bow = BagOfWords()
    enc = Encoder([fs, bow])

    suffix = seralize_dataset(data_dir, aut_pair, nfeatures, words_per_tile=words_per_tile,
                              vectoriser=enc, force=force, shrink_train=shrink_train, shrink_test=shrink_test)
    X_train, y_train, X_test, y_test = deseralize_dataset(suffix)

    models_with_names = (
        (RidgeClassifier(tol=1e-2, solver="sag", alpha=10000), "Ridge Classifier")
        , (Perceptron(n_iter=50, penalty='l2', alpha=0.001), "Perceptron")
        , (PassiveAggressiveClassifier(n_iter=50, C=1e-8), "Passive-Aggressive")
        , (SGDClassifier(alpha=.002, n_iter=50, penalty="elasticnet"), "LinearSVC with elastic-Net penalty")
        , (NearestCentroid(), "NearestCentroid (aka Rocchio classifier)")
        , (MultinomialNB(alpha=.01), "MultinomialNB")
        , (BernoulliNB(alpha=.01), "BernoulliNB")
        , (LinearSVC(penalty="l2", dual=False, tol=1e-3), "LinearSVC with l2 penalty")
        , (SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"), "SGDClassifier")
        , (LinearSVC(), "LinearSVC"))

    results = train_models(X_train, y_train, X_test, y_test, models_with_names)
    plot_results(results)


if __name__ == '__main__':
    main()
