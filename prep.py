import itertools as itt
import numpy as np

import io
import os
import pickle
from time import time
from random import shuffle

from feature_engineering import bag_of_words, BagOfWords


def build_tiles(text, nwords):
    ll = (s.strip().split() for s in text.splitlines())
    words = list(itt.chain(*ll))

    word_lists = (words[i:i+nwords] for i, _ in enumerate(words[:-(nwords-1)]))
    return [" ".join(wl) for wl in word_lists]


def break_on_tiles(class1file, class2file, words_per_tile, shrink=1):
    print("Opening files:\n%s\n%s" % (class1file, class2file))
    with io.open(class1file) as f1, \
            io.open(class2file) as f2:
        text1 = f1.read()
        text2 = f2.read()

    n1, n2 = len(text1) // shrink, len(text2) // shrink
    text1, text2 = text1[:n1], text2[:n2]

    print("Size of the first text = %d" % len(text1))
    print("Size of the second text = %d" % len(text2))

    t0 = time()
    tiles1 = build_tiles(text1, words_per_tile)
    print("broke text1 in tiles in %fs" % (time() - t0))

    t1 = time()
    tiles2 = build_tiles(text2, words_per_tile)
    print("broke text2 in tiles in %fs" % (time() - t1))

    # making equal
    min_samples = min(len(tiles1), len(tiles2))
    tiles1, tiles2 = tiles1[:min_samples], tiles2[:min_samples]

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


def seralize_dataset(data_dir, aut_pair, nfeatures, suffix, vectorised=True, words_per_tile=100, force=False,
                     shrink_train=1, shrink_test=1):

    if not force and all(map(os.path.isfile, ["serialized/X_train" + suffix + ".p",
                                              "serialized/y_train" + suffix + ".p",
                                              "serialized/X_test" + suffix + ".p",
                                              "serialized/y_test" + suffix + ".p"])):
        print("Serialized dataset have been found")
        return

    print("Serializing dataset " + suffix)
    train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    X_train, y_train = break_on_tiles(train_fn1, train_fn2, words_per_tile, shrink=shrink_train)

    test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")
    X_test, y_test = break_on_tiles(test_fn1, test_fn2, words_per_tile, shrink=shrink_test)

    if vectorised:
        bow = BagOfWords(nfeatures)
        X_train = bow.fit_transform(X_train, y_train)
        X_test = bow.transform(X_test)

    pickle.dump(X_train, open("serialized/X_train" + suffix + ".p", "wb"))
    pickle.dump(y_train, open("serialized/y_train" + suffix + ".p", "wb"))
    pickle.dump(X_test, open("serialized/X_test" + suffix + ".p", "wb"))
    pickle.dump(y_test, open("serialized/y_test" + suffix + ".p", "wb"))


def deseralize_dataset(suffix):
    t = time()
    with open("serialized/X_train" + suffix + ".p", 'rb') as X_train_f,\
            open("serialized/y_train" + suffix + ".p", 'rb') as y_train_f,\
            open("serialized/X_test" + suffix + ".p", 'rb') as X_test_f,\
            open("serialized/y_test" + suffix + ".p", 'rb') as y_test_f:
        X_train = pickle.load(X_train_f)
        y_train = pickle.load(y_train_f)
        X_test = pickle.load(X_test_f)
        y_test = pickle.load(y_test_f)
    print("Deserialization finished in {}s".format(time() - t))

    return X_train, y_train, X_test, y_test
