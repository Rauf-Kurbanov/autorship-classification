import pandas as pd

from collections import Counter


def punct_features(tokens, i):
    return {'next-word-capitalized': False if i == len(tokens) - 1 else tokens[i+1][0].isupper(),
            'prev-word-is-one-char': False if i == 0 else len(tokens[i-1]) == 1}


def tile_punct_features(tile):
    words = tile.split()
    feature_dict1 = [punct_features(words, i) for i in range(len(words))]

    feature_df = pd.DataFrame(feature_dict1)
    return feature_df.sum() / len(words)


def letter_features(name):
    name = name.lower()
    return Counter(name)


def tile_letter_features(tile, language):
    lang_to_alph = {"RUS": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                    "ENG": "abcdefghijklmnopqrstuvwxyz"}
    alphabet = lang_to_alph[language]

    tile = tile.lower()
    words = tile.split()

    tile_counter = Counter()
    for c in (letter_features(word) for word in words):
        tile_counter.update(c)

    return {k: tile_counter[k] / len(words) for k in alphabet}


def tile_features(tile, language):
    tlf = tile_letter_features(tile, language)
    tpf = tile_punct_features(tile)
    return list(tlf.values()) + list(tpf)