import os
import math
import numpy as np
import itertools as itt

from clean_test import prepare_training_set, vectorise_dataset, proportion, deseralize_dataset, seralize_dataset

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM


def simple_model(nfeatures):
    model = Sequential()
    model.add(Embedding(nfeatures, 256, input_length=nfeatures, init='uniform'))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def lstm_model(nfeatures):
    model = Sequential()
    model.add(Embedding(nfeatures, 256, input_length=nfeatures, init='uniform'))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def main():
    nfeatures = 1000
    batch_size = 32
    nb_epoch = 2
    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko"
    suffix = str(nfeatures) + aut_pair

    seralize_dataset(data_dir, aut_pair, nfeatures, suffix)
    X_train, y_train, X_test, y_test = deseralize_dataset(suffix)

    dlen = X_train.shape[0] // 10000
    dlen -= dlen % batch_size
    X_train, y_train = X_train[:dlen, :], y_train[:dlen]
    print("Cropped X_train.shape = {}".format(X_train.shape))


    tlen = X_test.shape[0] // 100
    X_test, y_test= X_test[:tlen, :], y_test[:tlen]
    print("X_test.shape = {}".format(X_test.shape))
    X_test = X_test.toarray()
    y_test = np.asarray(y_test)

    # model = lstm_model(nfeatures)
    model = simple_model(nfeatures)

    def batch_gen(batch_size, dlen):
        for i in range(0, dlen, batch_size):
            yield (X_train[i:i+batch_size, :].toarray(), np.asarray(y_train[i:i+batch_size]))

    sample_gen = itt.cycle(batch_gen(batch_size, dlen))
    model.fit_generator(sample_gen,
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, y_test),
                        samples_per_epoch=dlen)
    # model.save("serialized/simple.keras")

    # score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
    # print('Training score:', score)
    # print('Training accuracy:', acc)
    # pred = model.predict(x=X_train, batch_size=batch_size)
    # print("Classes proportion on train: {}".format(proportion(pred)))

    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    pred = model.predict(x=X_test, batch_size=batch_size)
    print("Classes proportion on test: {}".format(proportion(pred)))


if __name__ == '__main__':
    main()
