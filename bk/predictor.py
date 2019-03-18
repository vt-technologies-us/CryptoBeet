# CryptoCurrency Price Predictor base on LSTM Neural Network
# Author: @InBeet


import numpy as np  # working with data
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import utilities

import os


def create_datasets(dataset, sequence_length):
    dataset = np.reshape(dataset, (-1, 1))

    sequence_length += 1
    seq_dataset = []
    for i in range(len(dataset) - sequence_length):
        seq_dataset.append(dataset[i: i + sequence_length])

    seq_dataset = np.array(seq_dataset)

    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]

    return data_x, data_y


# print(device_lib.list_local_devices())

data_bank = 'CryptoAI-CoinMarketCapHistoricalDataScraper/data'
items = os.listdir(data_bank)
print(items)
raw_profit = [0.0] * len(items)
for idx, item in enumerate(items):
    currency = (item.split('.')[0])
    print(currency)

    currency_data = utilities.get_dataset(currency=currency)

    currency_close_price = currency_data.close.values.astype('float16')
    currency_close_price = currency_close_price.reshape(len(currency_close_price), 1)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    scaler = MinMaxScaler(feature_range=(0, 1))
    currency_close_price_scaled = scaler.fit_transform(currency_close_price)

    train_size = int(len(currency_close_price_scaled) * 0.85)
    test_size = len(currency_close_price_scaled) - train_size
    train, test = currency_close_price_scaled[0:train_size, :], currency_close_price_scaled[
                                                                train_size:len(currency_close_price_scaled), :]

    look_back = 10

    x_train, y_train = create_datasets(train, look_back)

    x_test, y_test = create_datasets(test, look_back)

    model = Sequential()

    model.add(LSTM(input_shape=(None, 1), units=100, return_sequences=True))
    model.add(Dropout(0.35))

    model.add(LSTM(150, return_sequences=False))
    model.add(Dropout(0.35))

    model.add(Dense(units=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    history = model.fit(x_train, y_train, batch_size=64, epochs=60, verbose=2, validation_split=0.2)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    train_predict_unnorm = scaler.inverse_transform(train_predict)
    test_predict_unnorm = scaler.inverse_transform(test_predict)

    # CREATING SIMILAR DATA-SET TO PLOT TRAINING PREDICTIONS
    trainPredictPlot = np.empty_like(currency_close_price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict_unnorm) + look_back, :] = train_predict_unnorm

    # CREATING SIMILAR DATA-SET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(currency_close_price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict_unnorm) + (look_back * 2) + 1:len(currency_close_price) - 1,
    :] = test_predict_unnorm

    # CALCULATE RAW PROFIT FOR EACH COIN

    p11 = y_test[-1][0]
    p10 = x_test[-1][0]
    profit = float((p11 - p10) / ((p11 + p10) / 2))
    print(profit)
    if profit > 0:
        raw_profit[idx] = profit

    plt.figure(figsize=(19, 10))
    plt.plot(currency_close_price, 'g', label='original dataset')
    plt.plot(trainPredictPlot, 'r', label='training set')
    plt.plot(testPredictPlot, 'b', label='predicted price/test set')
    plt.legend(loc='upper left')
    plt.xlabel('Time in Days')
    plt.ylabel('Price')

    plt.title("%s price %s - % s" % (currency,
                                     utilities.get_date_from_current(offset=len(currency_close_price)),
                                     utilities.get_date_from_current(0)))

    plt.figure(figsize=(19, 10))
    plt.plot(currency_close_price[-250:], 'g', label='original dataset')
    plt.plot(trainPredictPlot[-250:], 'r', label='training set')
    plt.plot(testPredictPlot[-250:], 'b', label='predicted price/test set')
    plt.legend(loc='upper left')
    plt.xlabel('Time in Days')
    plt.ylabel('Price')

    plt.title("%s price %s - % s" % (currency,
                                     utilities.get_date_from_current(offset=len(currency_close_price)),
                                     utilities.get_date_from_current(0)))

    plt.show()

# CALCULATE THE COINS PORTFOLIO SHARE LIST
s = sum(raw_profit)
portfolio_share = reversed(sorted(zip(raw_profit, items)))
print('Results:')
[print(item.split('.')[0], x / s) for x, item in portfolio_share]
