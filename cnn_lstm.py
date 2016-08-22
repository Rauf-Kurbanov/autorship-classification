from __future__ import print_function
import numpy as np
import os

from clean_test import prepare_training_set, vectorise_dataset

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D

# Embedding
max_features = 20000
maxlen = 1000
embedding_size = 128

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

data_dir = "/home/rauf/Programs/shpilman/data"
aut_pair = "perumov-vs-lukjarenko"
# aut_pair = "asimov-vs-silverberg"

train_fn1 = os.path.join(data_dir, aut_pair, "class1_training/class1_training.txt")
train_fn2 = os.path.join(data_dir, aut_pair, "class2_training/class2_training.txt")
X_train, y_train = prepare_training_set(train_fn1, train_fn2)

test_fn1 = os.path.join(data_dir, aut_pair, "class1_test/class1_test.txt")
test_fn2 = os.path.join(data_dir, aut_pair, "class2_test/class2_test.txt")

X_test, y_test = prepare_training_set(test_fn1, test_fn2)
X_train, X_test = vectorise_dataset(X_train, X_test, y_train, select_chi2=maxlen)
#
X_train = X_train.toarray()
y_train = np.asarray(y_train)

# dlen = 20000
# X_train, y_train = X_train[:dlen, :], y_train[:dlen]

X_test = X_test.toarray()
y_test = np.asarray(y_test)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)