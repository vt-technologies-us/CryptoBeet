# CryptoCurrency Price Predictor base on LSTM-GRU Neural Network
# Author: @VT-tech 

# Copyright 2018 The VT tech co. All Rights Reserved.
#
# Licensed under the Apache License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/vt-technologies-us/CryptoBeet/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Bring in all of the public InBeet interface into this
# module.

import tables
import shelve

import scipy.signal as sig
import numpy as np  # working with data
import pandas as pd

from tensorflow.python.client import device_lib

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utilities
import primal_trader

X_NUM = 3
Y_NUM = 4
gs = gridspec.GridSpec(X_NUM, Y_NUM)
fig_loss = plt.figure(figsize=(19, 10))
fig_tot = plt.figure(figsize=(19, 10))
fig_test = plt.figure(figsize=(19, 10))
fig_rates = plt.figure(figsize=(19, 10))


def create_datasets(dataset, sequence_length):
    sequence_length += 1
    seq_dataset = []
    for i in range(len(dataset) - sequence_length):
        seq_dataset.append(dataset[i: i + sequence_length])

    #    seq_dataset = np.array(seq_dataset)
    seq_dataset = np.array(seq_dataset, dtype=np.float16)

    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]

    return data_x, data_y


def plot_training_loss(history, i=0, j=0, **kwargs):
    currency = kwargs['currency'] if 'currency' in kwargs else ''
    ax = fig_loss.add_subplot(gs[i, j])
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('model loss ' + currency)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')


def plot_train_test(currency_close_price, trainPredictPlot, testPredictPlot, i, j, **kwargs):
    currency = kwargs['currency'] if 'currency' in kwargs else ''

    ax = fig_tot.add_subplot(gs[i, j])
    ax.plot(currency_close_price, 'g', label='original dataset')
    ax.plot(trainPredictPlot, 'r', label='training set')
    ax.plot(testPredictPlot, 'b', label='predicted price/test set')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time in Days')
    ax.set_ylabel('Price')
    ax.set_title("%s price" % (currency))

    ax = fig_test.add_subplot(gs[i, j])
    ax.plot(currency_close_price[-900:], 'g', label='original dataset')
    ax.plot(trainPredictPlot[-900:], 'r', label='training set')
    ax.plot(testPredictPlot[-900:], 'b', label='predicted price/test set')
    ax.legend(loc='upper left')
    ax.set_xlabel('Time in Days')
    ax.set_ylabel('Price')
    ax.set_title("%s price" % (currency))


def plot_raw_profit(raw_profit, i=0, j=0, **kwargs):
    currency = kwargs['currency'] if 'currency' in kwargs else ''
    ax = fig_rates.add_subplot(gs[i, j])
    ax.plot(raw_profit)
    ax.set_title('model loss ' + currency)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')


def learn(db, l=0, max_size=0):
    i = int(l / Y_NUM)
    j = l % Y_NUM
    currency = db.name
    print(l, currency)
    # x1 = db[:]['ticker'][:, 6]    # db[:]['trades'].shape, db[:]['book'].shape
    if max_size == 0:
        currency_close_price = db[:]['trades'][:, :, 3].mean(axis=1)
    else:
        currency_close_price = db[:max_size]['trades'][:, :, 3].mean(axis=1)
    for idx in np.where(currency_close_price == -1)[0]:
        currency_close_price[idx] = currency_close_price[idx - 1] * (1 + np.random.randn() * 0.0005)

    # currency_close_price = sig.medfilt(currency_close_price, 5)
    currency_close_price = currency_close_price.reshape((-1, 1))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    currency_close_price_scaled = scaler.fit_transform(currency_close_price)

    train_size = int(len(currency_close_price_scaled) * 0.85)
    test_size = len(currency_close_price_scaled) - train_size
    train, test = currency_close_price_scaled[0:train_size, :], currency_close_price_scaled[
                                                                train_size:len(currency_close_price_scaled), :]

    look_back = 10

    x_train, y_train = create_datasets(train, look_back)
    x_test, y_test = create_datasets(test, look_back)

    model = Sequential()

    # keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
    # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
    # unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    # activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    # bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,
    # return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

    model.add(LSTM(input_shape=(None, 1), units=100, recurrent_dropout=0.18, kernel_initializer='glorot_uniform',
                   return_sequences=True, use_bias=True))
    model.add(Dropout(0.17))

    model.add(
        LSTM(150, return_sequences=False, recurrent_dropout=0.17, kernel_initializer='glorot_uniform', use_bias=True,
             recurrent_activation='hard_sigmoid', ))
    model.add(Dropout(0.18))

    model.add(Dense(units=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    history = model.fit(x_train, y_train, batch_size=64, epochs=40, verbose=2, validation_split=0.25)

    plot_training_loss(history, i, j, currency=currency)

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    train_predict_unnorm = scaler.inverse_transform(train_predict)
    test_predict_unnorm = scaler.inverse_transform(test_predict)

    # del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one

    # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
    trainPredictPlot = np.empty_like(currency_close_price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict_unnorm) + look_back, :] = train_predict_unnorm

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(currency_close_price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict_unnorm) + (look_back * 2) + 1:len(currency_close_price) - 1, :] = \
        test_predict_unnorm

    plot_train_test(currency_close_price, trainPredictPlot, testPredictPlot, i, j, currency=currency)

    test_price = currency_close_price[len(train_predict_unnorm) + (look_back * 2) + 1:len(currency_close_price) - 1, :]
    rate = (test_predict_unnorm[1:] - test_price[:-1]) / test_price[:-1]

    return model, rate, test_price[:-1]


print(device_lib.list_local_devices())


def main():
    db_filename = './data/b.h5'
    models = {}
    rates = {}
    test_price = {}

    with tables.File(db_filename, 'r') as f:
        for l, db in enumerate(f.root):
            if l > 1:
                pass
                # continue
            models[db.name], rates[db.name], test_price[db.name] = learn(db, l)
            models[db.name].save('model/LSTM_model_{}.h5'.format(db.name))  # creates a HDF5 file 'my_model.h5'
            # model = load_model('model/LSTM_model_{}.h5'.format(db.name))

        for currency, raw_profit in rates.items():
            print(currency)
            plot_raw_profit(raw_profit, i=0, j=0, currency=currency)

            # Calculate the Coins PORTFOLIO Share List

            s = sum(raw_profit)
            portfolio_share = reversed(sorted(zip(raw_profit)))
            print('Results:')

        plt.show()

    workspace_output = 'models_5min.out'
    my_shelf = shelve.open(workspace_output, 'n')  # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))

    my_shelf.close()

    # CALCULATE THE COINS PORTFOLIO SHARE LIST
    # s = sum(raw_profit)
    # portfolio_share = reversed(sorted(zip(raw_profit, items)))
    # print('Results:')
    # [print(item.split('.')[0], x / s) for x, item in portfolio_share]


# model = load_model('my_model.h5')

if __name__ == '__main__':
    main()
