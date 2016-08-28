import itertools as itt
import numpy as np
from enum import Enum
from functools import reduce
import scipy.sparse
from scipy.sparse import csr_matrix


class FeatureLevel(Enum):
    word = 0
    tile = 1
    text = 2


class Feature:
    def __init__(self, name, feature_level, extract, combine=None):
        self.name = name
        self.feature_level = feature_level
        self.extract = extract
        self.combine = combine

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)


class FeatureSet:
    _n_features = 0
    _features = {}

    def __init__(self, features):
        for f in features:
            self._features[f.feature_level].append(f)

        self._n_features = len(features)

    def compile(self):
        fwf = self._super_words_feature()
        self._features[FeatureLevel.tile].append(fwf)
        self._n_features -= len(self._features[FeatureLevel.word]) - 1

    def transform_tile(self, tile):
        return [f.extract(tile) for f in self._features[FeatureLevel.tile]]

    def transform_text(self, text):
        return [f.extract(text) for f in self._features[FeatureLevel.text]]

    def add_feature(self, feature):
        self._features[feature.feature_level].append(feature)
        self._n_features += 1

    def _super_words_feature(self):
        word_features = self._features[FeatureLevel.word]
        combines = [wf.combine for wf in word_features]

        def combine(resA, resB):
            return (c(ra, rb) for ra, rb, c in zip(resA, resB, combines))

        def extract(word):
            return (wf(word) for wf in word_features)

        return Feature("super_word_feature", FeatureLevel.word, extract, combine)

    def _from_word_features(self):
        swf = self._super_words_feature()

        def extract(tile):
            ll = (s.strip().split() for s in tile.splitlines())
            words = list(itt.chain(*ll))

            return reduce(swf.combine, map(swf.extract, words))

        return Feature("super_from_word_feature", FeatureLevel.tile, extract)


def feature(level, combine=None):
    def wrapper(func):
        name = func.__name__
        return Feature(name, level, func, combine)

    return wrapper


class Encoder:

    def __init__(self, feature_set, words_per_tile=100):
        self._feature_set = feature_set
        self._feature_set.compile()
        self._words_per_tile = words_per_tile

    def encode(self, text):
        tiles = self.build_tiles(text, self._words_per_tile)

        trans_by_tiles = np.array(self._feature_set.transform_tile(tile) for tile in tiles)
        trans_by_text = self._feature_set.transform_text(text)

        transfomed = trans_by_text
        transfomed.append(trans_by_tiles)
        encoded = reduce(self.concat_encodings, transfomed)

        return encoded

    @staticmethod
    def build_tiles(text, nwords):
        ll = (s.strip().split() for s in text.splitlines())
        words = list(itt.chain(*ll))

        word_lists = (words[i:i + nwords] for i, _ in enumerate(words[:-(nwords - 1)]))
        return (" ".join(wl) for wl in word_lists)

    @staticmethod
    def concat_encodings(x, y):
        if isinstance(x, csr_matrix) or isinstance(y, csr_matrix):
            return scipy.sparse.hstack((x, y))

        return np.hstack((x, y))


def main():
    @feature(FeatureLevel.word, lambda x, y: x + y)
    def word_length(word):
        return len(word)

    f = word_length
    print(type(f))

if __name__ == '__main__':
    main()
