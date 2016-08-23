import os
import numpy as np

from clean_test import prepare_training_set, vectorise_dataset, proportion, deseralize_dataset

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM


def simple_model(inp_layer):
    model = Sequential()
    model.add(Embedding(inp_layer, 256, input_length=inp_layer, init='uniform'))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def lstm_model(inp_layer):
    model = Sequential()
    model.add(Embedding(inp_layer, 256, input_length=inp_layer, init='uniform'))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def main():
    inp_layer = 1000
    batch_size = 32
    nb_epoch = 2

    data_dir = "/home/rauf/Programs/shpilman/data"
    aut_pair = "perumov-vs-lukjarenko"
    # aut_pair = "asimov-vs-silverberg"

    train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
    train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
    X_train, y_train = prepare_training_set(train_fn1, train_fn2)

    test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
    test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")

    X_test, y_test = prepare_training_set(test_fn1, test_fn2)
    X_train, X_test = vectorise_dataset(X_train, X_test, y_train, select_chi2=inp_layer)
    #
    # X_train, y_train, X_test, y_test = deseralize_dataset()

    # Temporary data subsampling
    dlen = X_train.shape[0] // 1000
    X_train, y_train = X_train[:dlen, :], y_train[:dlen]
    print("X_train.shape = {}".format(X_train.shape))

    tlen = X_test.shape[0] // 10
    X_test, y_test= X_test[:dlen, :], y_test[:dlen]
    print("X_test.shape = {}".format(X_train.shape))
    X_test = X_test.toarray()
    y_test = np.asarray(y_test)

    # model = lstm_model(inp_layer)
    model = simple_model(inp_layer)
    # model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)

    #TODO debug generator
    def batch_gen(batch_size, dlen):
        for i in range(0, dlen, batch_size):
            yield (X_train[i:i+batch_size, :].toarray(), np.asarray(y_train[i:i+batch_size]))

    model.fit_generator(batch_gen(batch_size, dlen),
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, y_test),
                        samples_per_epoch=dlen)
    model.save("serialized/lstm.keras")


    score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
    print('Training score:', score)
    print('Training accuracy:', acc)
    pred = model.predict(x=X_train, batch_size=batch_size)
    print("Classes proportion on train: {}".format(proportion(pred)))

    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    pred = model.predict(x=X_test, batch_size=batch_size)
    print("Classes proportion on test: {}".format(proportion(pred)))


if __name__ == '__main__':
    main()
