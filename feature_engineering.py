import pandas as pd
import itertools as itt
import numpy as np

from collections import Counter

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2

from feature_set import feature, FeatureLevel, Feature, FeatureSet, word_feature

import nltk
from nltk.corpus import names, brown
import random

from sklearn.feature_extraction import DictVectorizer
import pandas as pd

class BagOfWords(FeatureSet):
    def __init__(self, select_chi2=1000):
        n_features = 2 ** 16
        self._select_chi2 = select_chi2
        hasher = HashingVectorizer(stop_words='english',
                                   non_negative=True,
                                   n_features=n_features)

        self._vectoriser = make_pipeline(hasher, TfidfTransformer(), SelectKBest(chi2, k=select_chi2))
        self.fit = self._vectoriser.fit
        self.transform = self._vectoriser.transform
        self.fit_transform = self._vectoriser.fit_transform

    def __str__(self):
        return "BagOfWords_shci={}".format(self._select_chi2)


class LettersCountFeature(Feature):

    feature_level = FeatureLevel.tile
    name = "LettersFeature"

    def __init__(self, language):
        self._language = language

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


class GenderFeature(Feature):
    name = "GenderFeature"
    feature_level = FeatureLevel.word

    def __init__(self):
        labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                         [(name, 'female') for name in names.words('female.txt')])
        random.shuffle(labeled_names)

        train_set = [(self._last_letter(n), gender) for (n, gender) in labeled_names]
        self._clf = nltk.NaiveBayesClassifier.train(train_set)

    def combine(self, x, y):
        return x + y

    def extract(self, word):
        return 1 if self._clf.classify(self._gender_features(word)) == 'male' else 0

    def _gender_features(self, word):
        return {'suffix1': word[-1:],
                'suffix2': word[-2:]}


def n_most_common_suff(n):
    suffix_fdist = nltk.FreqDist()
    for word in brown.words():
        word = word.lower()
        suffix_fdist[word[-1:]] += 1
        suffix_fdist[word[-2:]] += 1
        suffix_fdist[word[-3:]] += 1

    return [suffix for (suffix, count) in suffix_fdist.most_common(n)]


class SuffixFeature(Feature):
    name = "Suffix"
    feature_level = FeatureLevel.word

    def __init__(self):
        self._common_suffixes = n_most_common_suff(100)

    def extract(self, word):
        return np.array([word.lower().endswith(suffix) for suffix in self._common_suffixes])

    def combine(self, x, y):
        return x + y


class PartOfSpeechFeature(Feature):
    name = "PartOfSpeechFeature"
    feature_level = FeatureLevel.word
    _common_suffixes = n_most_common_suff(100)

    def __init__(self):
        tagged_words = brown.tagged_words(categories='news')
        train_set = [(self._pos_features(n), g) for (n, g) in tagged_words]
        self._clf = nltk.DecisionTreeClassifier.train(train_set)

        _, labels = zip(*train_set)
        labels = [[x] for x in labels]
        labels = np.array(labels)
        self._dv = DictVectorizer(sparse=True)
        df = pd.DataFrame(labels).convert_objects(convert_numeric=True)
        self._dv.fit(df.to_dict(orient='records'))

    def _pos_features(self, word):
        features = {}
        for suffix in self._common_suffixes:
            features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
        return features

    def extract(self, word):
        label =  self._clf.classify(self._pos_features(word))
        df = pd.DataFrame([label]).convert_objects(convert_numeric=True)
        return self._dv.transform(df.to_dict(orient='records')).toarray()

    def combine(self, x, y):
        return x + y


def get_words(text):
    ll = (s.strip().split() for s in text.splitlines())
    return list(itt.chain(*ll))


@feature(FeatureLevel.tile)
def has_letter(tile):
    lcf = LettersCountFeature()
    return [n > 0 for n in lcf.extract(tile)]


@feature(FeatureLevel.tile)
def next_word_capitalized(tile):
    def next_cap(tokens, i):
        return False if i == len(tokens) - 1 else tokens[i+1][0].isupper()

    words = get_words(tile)
    mask = np.array([next_cap(words, i) for i, _ in enumerate(words)])
    return sum(mask) / len(words)


@feature(FeatureLevel.tile)
def prev_word_is_one_char(tile):
    def next_cap(tokens, i):
        return False if i == 0 else len(tokens[i-1]) == 1

    words = get_words(tile)
    mask = np.array([next_cap(words, i) for i, _ in enumerate(words)])
    return sum(mask) / len(words)


@feature(FeatureLevel.tile)
def tile_punct_features(tile):
    def punct_features(tokens, i):
        return [False if i == len(tokens) - 1 else tokens[i + 1][0].isupper(),
                False if i == 0 else len(tokens[i - 1]) == 1]

    words = list(get_words(tile))
    feature_dict1 = [punct_features(words, i) for i in range(len(words))]

    feature_df = pd.DataFrame(feature_dict1)
    return np.array(feature_df.sum() / len(words))


@word_feature(combine=lambda x, y: x + y)
def word_length(word):
    return len(word)


@word_feature(combine=lambda x, y: x + y)
def capitalized(word):
    return word[0].isupper()

