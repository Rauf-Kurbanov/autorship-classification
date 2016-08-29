from feature_encoder import FeatureSet, concat_encodings
from feature_engineering import tile_punct_features, LettersFeature, BagOfWords, word_length
from prep import seralize_dataset, deseralize_dataset
from test import train_all_models, plot_results


def main():
    language = "RUS"

    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko_mini"
    nfeatures = 1000
    words_per_tile = 100
    vectorised = False
    force = True
    shrink_train = 10
    shrink_test = 10

    suffix = "{}_nf={}_wpt={}_vec={}_shTr={}_shTst={}".format(aut_pair, nfeatures, words_per_tile, vectorised,
                                                              shrink_train, shrink_test)
    seralize_dataset(data_dir, aut_pair, nfeatures, suffix=suffix, words_per_tile=words_per_tile,
                     vectorised=vectorised, force=force, shrink_train=shrink_train, shrink_test=shrink_test)
    X_train, y_train, X_test, y_test = deseralize_dataset(suffix)

    fs = FeatureSet([tile_punct_features, LettersFeature(language), word_length])
    bow = BagOfWords()
    X_train1 = fs.fit_transform(X_train)
    X_test1 = fs.transform(X_test)

    X_train2 = bow.fit_transform(X_train, y_train)
    X_test2 = bow.transform(X_test)

    X_train = concat_encodings(X_train1, X_train2)
    X_test = concat_encodings(X_test1, X_test2)

    results = train_all_models(X_train, y_train, X_test, y_test)
    plot_results(results)


if __name__ == '__main__':
    main()
