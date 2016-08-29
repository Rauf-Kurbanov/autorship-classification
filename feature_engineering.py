import pandas as pd
import itertools as itt
import numpy as np

from collections import Counter

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2

from feature_encoder import feature, FeatureLevel, Feature, FeatureSet, word_feature


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
    assert(True)
    feature_dict1 = [punct_features(words, i) for i in range(len(words))]

    feature_df = pd.DataFrame(feature_dict1)
    return np.array(feature_df.sum() / len(words))


@word_feature(combine=lambda x, y: x + y)
def word_length(word):
    return len(word)


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


class LettersFeature(Feature):
    def __init__(self, language):
        self._language = language
        self.feature_level = FeatureLevel.tile

    def extract(self, tile):
        lang_to_alph = {"RUS": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                        "ENG": "abcdefghijklmnopqrstuvwxyz"}
        alphabet = lang_to_alph[self._language]
        tile = tile.lower()
        words = tile.split()

        tile_counter = Counter()
        for c in (Counter(word.lower()) for word in words):
            tile_counter.update(c)

        return [tile_counter[k] / len(words) for k in alphabet]


class BagOfWords(FeatureSet):
    def __init__(self, select_chi2=1000):
        n_features = 2 ** 16
        hasher = HashingVectorizer(stop_words='english',
                                   non_negative=True,
                                   n_features=n_features)

        self._vectoriser = make_pipeline(hasher, TfidfTransformer(), SelectKBest(chi2, k=select_chi2))
        self.fit = self._vectoriser.fit
        self.transform = self._vectoriser.transform
        self.fit_transform = self._vectoriser.fit_transform
