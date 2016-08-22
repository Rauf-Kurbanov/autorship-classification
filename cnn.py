from __future__ import print_function

import os
import numpy as np

from clean_test import prepare_training_set, vectorise_dataset

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras import backend as K


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
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
X_train, X_test = vectorise_dataset(X_train, X_test, y_train, select_chi2=400)
#
X_train = X_train.toarray()
y_train = np.asarray(y_train)

dlen = 20000
X_train, y_train = X_train[:dlen, :], y_train[:dlen]

X_test = X_test.toarray()
y_test = np.asarray(y_test)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model.add(MaxPooling1D(pool_length=model.output_shape[1]))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))