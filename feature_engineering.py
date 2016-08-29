import pandas as pd
import itertools as itt
import numpy as np

from collections import Counter

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2

from feature_encoder import feature, FeatureLevel, Feature


def get_words(text):
    ll = (s.strip().split() for s in text.splitlines())
    return list(itt.chain(*ll))


# next-word-capitalized
# prev-word-is-one-char
def punct_features(tokens, i):
    return [False if i == len(tokens) - 1 else tokens[i+1][0].isupper(),
            False if i == 0 else len(tokens[i-1]) == 1]


@feature(FeatureLevel.tile)
def tile_punct_features(tile):
    words = list(get_words(tile))
    feature_dict1 = [punct_features(words, i) for i in range(len(words))]

    feature_df = pd.DataFrame(feature_dict1)
    return np.array(feature_df.sum() / len(words))


@feature(FeatureLevel.tile)
def tile_letter_features(language, tile):
    lang_to_alph = {"RUS": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    "ENG": "abcdefghijklmnopqrstuvwxyz"}
    alphabet = lang_to_alph[language]

    tile = tile.lower()
    words = tile.split()

    tile_counter = Counter()
    for c in (Counter(word.lower()) for word in words):
        tile_counter.update(c)

    return [tile_counter[k] / len(words) for k in alphabet]


def bag_of_words(X_train, X_test, y_train, select_chi2=1000):
    n_features = 2 ** 16

    hasher = HashingVectorizer(stop_words='english',
                               non_negative=True,
                               n_features=n_features)

    vectoriser = make_pipeline(hasher, TfidfTransformer(), SelectKBest(chi2, k=select_chi2))

    X_train = vectoriser.fit_transform(X_train, y_train)
    X_test = vectoriser.transform(X_test)

    return X_train, X_test
