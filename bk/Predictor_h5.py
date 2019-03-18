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
import time

import keras
import tables
import shelve

import scipy.signal as sig
import numpy as np  # working with data
import pandas as pd
import theano.tensor as T

from tensorflow.python.client import device_lib

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import GRU, LSTM
from keras.models import Sequential
from keras.models import load_model

from itosfm import ITOSFM

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utilities
import primal_trader

Mode = 'LSTM'
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


def build_fsm_model(layers, freq, learning_rate):
    model = Sequential()

    model.add(ITOSFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        freq_dim=freq,
        return_sequences=True))

    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print("Compilation Time : ", time.time() - start)
    return model


def make_model(mode='LSTM'):
    model = Sequential()

    if mode == 'GRU':

        # keras.layers.GRU(units, activation='tanh',
        # recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
        # recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None,
        # recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        # recurrent_constraint=None, bias_constraint=None, dropout=0.0,
        # recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False,
        # stateful=False, unroll=False, reset_after=False)

        model.add(GRU(input_shape=(None, 1), units=100, return_sequences=True, recurrent_dropout=0.35, use_bias=True, ))
        # model.add(Dropout(0.18))

        model.add(GRU(150, return_sequences=False, recurrent_dropout=0.35, use_bias=True, ))
        # model.add(Dropout(0.18))

        model.add(Dense(units=1))

        if mode == 'SFM':
            model = Sequential()
            # model.add(ITOSFM(input_dim=layers[0], hidden_dim=layers[1], output_dim=layers[2], freq_dim=freq, return_sequences=True))

    elif mode == 'LSTM':
        # keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
        # kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
        # unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
        # activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
        # bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1,
        # return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

        model.add(LSTM(input_shape=(None, 1), units=100, recurrent_dropout=0.18, kernel_initializer='glorot_uniform',
                       return_sequences=True, use_bias=True))
        model.add(Dropout(0.17))

        model.add(LSTM(150, return_sequences=False, recurrent_dropout=0.17, kernel_initializer='glorot_uniform',
                       use_bias=True, recurrent_activation='hard_sigmoid', ))

        model.add(Dropout(0.18))

        model.add(Dense(units=1))

    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model


def dataset_remove_null_values(db, max_size):
    # x1 = db[:]['ticker'][:, 6]    # db[:]['trades'].shape, db[:]['book'].shape

    if max_size == 0:
        currency_close_price = db[:]['trades'][:, :, 3].mean(axis=1)
    else:
        currency_close_price = db[:max_size]['trades'][:, :, 3].mean(axis=1)

    for idx in np.where(currency_close_price == -1)[0]:
        currency_close_price[idx] = currency_close_price[idx - 1] * (1 + np.random.randn() * 0.0005)

    # currency_close_price = sig.medfilt(currency_close_price, 5)
    currency_close_price = currency_close_price.reshape((-1, 1))

    return currency_close_price


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


def make_train_test(currency_close_price, look_back):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    currency_close_price_scaled = scaler.fit_transform(currency_close_price)

    train_size = int(len(currency_close_price_scaled) * 0.85)
    test_size = len(currency_close_price_scaled) - train_size
    train, test = currency_close_price_scaled[0:train_size, :], currency_close_price_scaled[
                                                                train_size:len(currency_close_price_scaled), :]

    x_train, y_train = create_datasets(train, look_back)
    x_test, y_test = create_datasets(test, look_back)

    return scaler, x_train, y_train, x_test, y_test


def make_result_model(model, scaler, currency_close_price, x_train, x_test, look_back):
    train_predict, test_predict = model.predict(x_train), model.predict(x_test)
    train_predict_unnorm, test_predict_unnorm = scaler.inverse_transform(train_predict), \
                                                scaler.inverse_transform(test_predict)

    # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
    train_predict_plot = np.empty_like(currency_close_price)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict_unnorm) + look_back, :] = train_predict_unnorm

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    test_predict_plot = np.empty_like(currency_close_price)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict_unnorm) + (look_back * 2) + 1:len(currency_close_price) - 1, :] = \
        test_predict_unnorm

    test_price = currency_close_price[len(train_predict_unnorm) + (look_back * 2) + 1:len(currency_close_price) - 1, :]
    rate = (test_predict_unnorm[1:] - test_price[:-1]) / test_price[:-1]

    return train_predict_plot, test_predict_plot, rate, test_price


def learn(db, l=0, max_size=0, mode='LSTM'):
    i = int(l / Y_NUM)
    j = l % Y_NUM
    currency = db.name
    print(l, currency)

    look_back = 10
    currency_close_price = dataset_remove_null_values(db, max_size)
    scaler, x_train, y_train, x_test, y_test = make_train_test(currency_close_price, look_back)

    model = make_model(mode)
    history = model.fit(x_train, y_train, batch_size=64, epochs=4, verbose=2, validation_split=0.25)

    plot_training_loss(history, i, j, currency=currency)

    train_predict_plot, test_predict_plot, rate, test_price = make_result_model(model, scaler, currency_close_price,
                                                                                x_train, x_test, look_back)

    plot_train_test(currency_close_price, train_predict_plot, test_predict_plot, i, j, currency=currency)

    return model, rate, test_price[:-1]


def main(mode='GRU'):
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
            models[db.name].save('model/{}_model_{}.h5'.format(mode, db.name))  # creates a HDF5 file 'my_model.h5'
            # model = load_model('model/{}_model_{}.h5'.format(mode, db.name))

        # del model  # deletes the existing model
        # returns a compiled model
        # identical to the previous one
        for currency, raw_profit in rates.items():
            print(currency)
            plot_raw_profit(raw_profit, i=0, j=0, currency=currency)

            # Calculate the Coins PORTFOLIO Share List

            s = sum(raw_profit)
            portfolio_share = reversed(sorted(zip(raw_profit)))
            print('Results:')

        plt.show()


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    main(Mode)
