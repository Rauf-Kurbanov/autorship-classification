import itertools as itt
import numpy as np
from enum import Enum
from functools import reduce
import scipy.sparse
from scipy.sparse import csr_matrix
from collections import defaultdict
import collections


class FeatureLevel(Enum):
    word = 0
    tile = 1
    text = 2


class Feature:
    def __init__(self, name, feature_level, extract, combine):
        self.name = name
        self.feature_level = feature_level
        self.extract = extract
        self.combine = combine

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)


class FeatureSet:
    _n_features = 0
    _features = defaultdict(list)

    def __init__(self, features):
        for f in features:
            self._features[f.feature_level].append(f)

        self._n_features = len(features)

    def __str__(self):
        return "{}".format([f.name for flist in self._features.values() for f in flist])

    def fit(self, X_train=None, y_train=None):
        swf = self._super_words_feature()
        self._features[FeatureLevel.tile].append(swf)
        self._n_features -= len(self._features[FeatureLevel.word]) - 1

    def transform(self, X_train):
        return np.array([self._transform_tile(x) for x in X_train])

    def fit_transform(self, X_train, y_train=None):
        self.fit(X_train, y_train)
        return self.transform(X_train)

    def add_feature(self, feature):
        self._features[feature.feature_level].append(feature)
        self._n_features += 1

    def _transform_tile(self, tile):
        return list(self.flatten(itt.chain(f.extract(tile) for f in self._features[FeatureLevel.tile])))

    def _super_words_feature(self):
        word_features = self._features[FeatureLevel.word]
        combines = [wf.combine for wf in word_features]

        def combine(resA, resB):
            return [c(ra, rb) for ra, rb, c in zip(resA, resB, combines)]

        def extract(word):
            return [wf(word) for wf in word_features]

        return Feature("super_word_feature", FeatureLevel.word, extract, combine)

    def _from_word_features(self):
        swf = self._super_words_feature()

        def extract(tile):
            ll = (s.strip().split() for s in tile.splitlines())
            words = list(itt.chain(*ll))

            return reduce(swf.combine, map(swf.extract, words))

        return Feature("super_from_word_feature", FeatureLevel.tile, extract)

    @staticmethod
    def flatten(l):
        for el in l:
            if isinstance(el, collections.Iterable):
                for sub in FeatureSet.flatten(el):
                    yield sub
            else:
                yield el


class Encoder(FeatureSet):
    def __init__(self, feature_sets):
        self._feature_sets = feature_sets

    def __str__(self):
        return "Encoder_fss={}".format([fs.__str__() for fs in self._feature_sets])

    def fit(self, X_train=None, y_train=None):
        for fs in self._feature_sets:
            fs.fit(X_train, y_train)

    def transform(self, X_train=None, y_train=None):
        transformed = [fs.transform(X_train) for fs in self._feature_sets]
        return reduce(self.concat_encodings, transformed)

    @staticmethod
    def concat_encodings(x, y):
        if isinstance(x, csr_matrix) or isinstance(y, csr_matrix):
            return scipy.sparse.hstack((x, y))

        return np.hstack((x, y))


def feature(level, combine=None):
    def wrapper(func):
        name = func.__name__
        return Feature(name, level, func, combine)

    return wrapper


def word_feature(combine=None):
    def wrapper(func):
        name = func.__name__
        return Feature(name, FeatureLevel.word, func, combine)

    return wrapper


def main():
    @feature(FeatureLevel.word, lambda x, y: x + y)
    def word_length(word):
        return len(word)

    f = word_length
    print(type(f))

if __name__ == '__main__':
    main()
