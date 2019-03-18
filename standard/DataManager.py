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
import datetime
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from sklearn.preprocessing import MinMaxScaler

from standard import config


class DataManager:
    _USD_book = np.array([[1, 1, np.inf], [1, 1, -np.inf]])

    def __init__(self, **kwargs):
        self.start_date = kwargs.get('start_date', None)
        self.end_date = kwargs.get('end_date', None)
        self.max_size_of_database = kwargs.get('max_size_of_database', np.inf)

        self._price_or_buy_sell = kwargs.get('price_or_buy_sell', 'buy_sell')
        self._plot_from_last = kwargs.get('plot_from_last', 1000)
        self._train_test_ratio = kwargs.get('train_test_ratio', 0.85)

        self.look_back = kwargs.get('look_back', 10)
        self.stride = kwargs.get('stride', config.stride)
        self.bad_coins = kwargs.get('bad_coins', config.bad_coins)

        # self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def initialize(self, file_address):
        self._load_databases(file_address)
        self._remove_null_data()
        self._scale_data()
        self._prepare_x_y()

    def _load_databases(self, file_address):
        with tables.File(file_address) as f:
            self.coins = list(f.root._v_children)
            for coin in self.bad_coins:
                if coin in self.coins:
                    self.coins.remove(coin)
            # shapes = list(x.shape[0] for x in f.root)
            max_len_coin = max(self.coins, key=lambda x: getattr(f.root, x).shape[0])
            max_len = min(getattr(f.root, max_len_coin).shape[0], self.max_size_of_database)

            # only manage between start time and end time
            data_coins = {}

            # todo one of them
            if self.start_date and self.end_date:
                period = getattr(f.root, max_len_coin).get_where_list(
                    f'(timestamp > {self.start_date.timestamp()}) & (timestamp < {self.end_date.timestamp()})')
                drop_coins = []
                for coin in self.coins:
                    p_coin = getattr(f.root, coin).get_where_list(
                        f'(timestamp > {self.start_date.timestamp()}) & (timestamp < {self.end_date.timestamp()})')
                    if p_coin.shape[0] > 0:
                        data_coins[coin] = getattr(f.root, coin)[p_coin]
                    else:
                        drop_coins.append(coin)
                for coin in drop_coins:
                    self.coins.remove(coin)
            else:
                period = np.arange(min(max_len, self.max_size_of_database))
                for coin in self.coins:
                    p_coin = np.arange(min(getattr(f.root, coin).shape[0], self.max_size_of_database))
                    data_coins[coin] = getattr(f.root, coin)[p_coin]

            self.num_coins = len(self.coins)

            if np.isfinite(self.max_size_of_database):
                period = period[:self.max_size_of_database]

            times = getattr(f.root, max_len_coin)[period]['timestamp']
            self._book_orders = {c: np.full((times.shape[0], 50, 3), np.nan) for c in self.coins}

            if self._price_or_buy_sell == 'price':
                data = np.full((times.shape[0], self.num_coins), np.nan)
            elif self._price_or_buy_sell == 'buy_sell':
                data = np.full((times.shape[0], self.num_coins * 2), np.nan)

            indices = [0] * self.num_coins
            for i in range(times.shape[0]):
                for k in range(len(indices)):
                    # todo use all data
                    while times[i] - data_coins[self.coins[k]][indices[k]]['timestamp'] > 150:
                        indices[k] += 1
                    j = indices[k]
                    if abs(times[i] - data_coins[self.coins[k]][j]['timestamp']) < 150:
                        if self._price_or_buy_sell == 'price':
                            data[i, k] = data_coins[self.coins[k]][j]['ticker'][6]
                        elif self._price_or_buy_sell == 'buy_sell':
                            data[i, 2 * k] = data_coins[self.coins[k]][j]['book'][0, 0]
                            data[i, 2 * k + 1] = data_coins[self.coins[k]][j]['book'][25, 0]

                        self._book_orders[self.coins[k]][i] = data_coins[self.coins[k]][j]['book']
                        indices[k] += 1

            if self._price_or_buy_sell == 'price':
                cols = self.coins
            elif self._price_or_buy_sell == 'buy_sell':
                cols = pd.MultiIndex.from_tuples(zip(np.repeat(self.coins, 2), ('buy', 'sell') * 2 * len(self.coins)))
            times = np.array([pd.Timestamp(datetime.datetime.fromtimestamp(t)) for t in times])

        self.df = pd.DataFrame(data, columns=cols, index=times)

    def _remove_null_data(self):
        drop_columns = []
        for c in self.df:
            if self._price_or_buy_sell == 'price':
                if np.isfinite(self.df[c].values[0]):
                    drop_columns.append(c)
            elif self._price_or_buy_sell == 'buy_sell':
                if not all(np.isfinite(self.df[c[0]].values[0])):
                    drop_columns.append(c)

        # mask = [c not in np.unique(np.asarray(drop_columns)[:, 0]) for c in self.coins]
        # self._book_orders = self._book_orders[mask]
        if drop_columns:
            self.df = self.df.drop(columns=drop_columns)
            for drop_coin in np.unique(np.asarray(drop_columns)[:, 0]):
                self.coins.remove(drop_coin)
                self._book_orders.pop(drop_coin, None)

        for coin in self.coins:
            x = self.df[coin].values
            for idx in np.unique(np.where(~np.isfinite(x))[0]):
                x[idx] = x[idx - 1] * (1 + np.random.randn() * 0.0005)
            for idx in np.unique(np.where(x < 0)[0]):  # change == -1 to < 0
                x[idx] = x[idx - 1] * (1 + np.random.randn() * 0.0005)
            self.df[coin] = x

    def _scale_data(self):
        self.scalers = {c: MinMaxScaler(feature_range=(-1, 1)) for c in self.coins}
        self.df_scaled = self.df.copy()
        for coin, sc in self.scalers.items():
            x = self.df_scaled[coin].values
            x_scaled = sc.fit_transform(x)
            self.df_scaled[coin] = x_scaled

    def _scale_new_data(self):
        self.df_scaled = self.df.copy()
        for coin, sc in self.scalers.items():
            x = self.df_scaled[coin].values
            x_scaled = sc.transform(x)
            self.df_scaled[coin] = x_scaled

    def _prepare_x_y(self):
        self.ds = {c: None for c in self.coins}
        for coin in self.ds:
            self.ds[coin] = self.create_datasets(self.df_scaled[coin].values)

    def create_datasets(self, dataset):
        sequence_length = self.look_back * self.stride + 1
        seq_dataset = []
        for i in range(len(dataset) - sequence_length + 1):
            seq_dataset.append(dataset[i: i + sequence_length: self.stride])

        #    seq_dataset = np.array(seq_dataset)
        seq_dataset = np.array(seq_dataset, dtype=np.float16)

        data_x = seq_dataset[:, :-1]
        data_y = seq_dataset[:, -1]

        return data_x, data_y

    def _prepare_x_y_test(self, len):
        ds = {c: None for c in self.coins}
        for coin in ds:
            dataset = self.df_scaled[coin].values[-self.look_back * self.stride - len + 1:]
            dataset = np.concatenate((dataset, np.full((len, dataset.shape[1]), np.nan)))
            ds[coin] = self.create_datasets(dataset)
        return ds

    def get_book_orders(self, time):
        book_orders = dict()
        for c in self._book_orders:
            book_orders[c] = self._book_orders[c][time]
        return book_orders

    def new_data(self, times, book_orders):
        if not isinstance(times, list):
            times = [times]
            for coin in book_orders:
                book_orders[coin] = np.expand_dims(book_orders[coin], axis=0)

        times = np.array([t if isinstance(times[0], pd.Timestamp) else pd.Timestamp(t * 1e9) for t in times])
        new_len = times.shape[0]

        data = {}
        for coin, book_order in self._book_orders.items():
            self._book_orders[coin] = np.concatenate((self._book_orders[coin], book_orders[coin]), axis=0)
            data[coin, 'buy'] = book_orders[coin][:, 0, 0]
            data[coin, 'sell'] = book_orders[coin][:, 25, 0]
            # data[coin] = book_orders[coin][:, [0, 25], 0]

        new_df = pd.DataFrame(data, index=times)

        # remove null datas
        for coin in self.coins:
            x = new_df[coin].values
            for idx in np.where(~np.isfinite(x))[0]:
                if idx == 0:
                    x[idx] = self.df[coin].values[-1] * (1 + np.random.randn() * 0.0005)
                else:
                    x[idx] = x[idx - 1] * (1 + np.random.randn() * 0.0005)
            for idx in np.where(x == -1)[0]:
                if idx == 0:
                    x[idx] = self.df[coin].values[-1] * (1 + np.random.randn() * 0.0005)
                else:
                    x[idx] = x[idx - 1] * (1 + np.random.randn() * 0.0005)
            new_df[coin] = x

        self.df = self.df.append(new_df)

        # Scale Data
        new_df_scaled = new_df.copy()
        for coin, sc in self.scalers.items():
            x = new_df_scaled[coin].values
            x_scaled = sc.transform(x)
            new_df_scaled[coin] = x_scaled
        self.df_scaled = self.df_scaled.append(new_df_scaled)

        # create new dataset
        return self._prepare_x_y_test(new_len)

    def get_train(self, coin):
        train_len = int(self._train_test_ratio * self.df[coin].shape[0])
        # return self.ds[coin][0][:train_len], self.ds[coin][1][:train_len]
        return {
            't': self.df.index[self.look_back * self.stride:train_len + self.look_back * self.stride],
            'x': self.ds[coin][0][:train_len],
            'y': self.ds[coin][1][:train_len]}

    def get_test(self, coin):
        train_len = int(self._train_test_ratio * self.df[coin].shape[0])
        return {
            't': self.df.index[train_len + self.look_back * self.stride:],
            'x': self.ds[coin][0][train_len:],
            'y': self.ds[coin][1][train_len:]}

    def plot_train_test(self, coin, ax, bounded=False):
        if bounded:
            train_len = min(int(self._train_test_ratio * self.df.shape[0]), self._plot_from_last)
        else:
            train_len = int(self._train_test_ratio * self.df.shape[0])
        data = self.df[coin].values
        times = self.df.index
        train_data = np.full(data.shape, np.nan)
        test_data = np.full(data.shape, np.nan)
        train_data[:train_len] = data[:train_len]
        test_data[train_len:] = data[train_len:]
        ax.plot(times, train_data, 'r', label='training set')
        ax.plot(times, test_data, 'b', label='predicted price/test set')

        ax.legend(loc='upper left')
        # ax.set_xlabel('Time in 5 minutes')
        ax.set_ylabel('Price')
        # ax.set_xlim(0, 1000)

    def plot_trends(self, ax, bounded=False):
        if bounded:
            plot_len = min(self.df.shape[0], self._plot_from_last)
        else:
            plot_len = self.df.shape[0]

        for coin in self.coins:
            data = self.df[coin].values / self.df[coin].values[0] * 100
            line = ax.plot(self.df.index, data, label=coin)
            plt.setp(line, linewidth=.5)

        line = ax.plot(self.df.index, np.full(self.df.index.shape, 100), label='USD')
        plt.setp(line, linewidth=.5)

        ax.legend(loc='upper left')
        # ax.set_xlabel('Time in 5 minutes')
        ax.set_ylabel('Price')
        # ax.set_xlim(0, 1000)

        return ax

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def save_for_tarding(self, filename):
        train_len = int(self._train_test_ratio * self.df.shape[0])
        self.save_scalers(filename)

        with open('models/start_trading_time.json', 'w') as f:
            json.dump(self.df.index[train_len - 1].isoformat(), f)
        config.start_date_test = self.df.index[train_len - 1]

        save_data = dict()
        save_data['dataframe'] = self.df[train_len - self.look_back * self.stride - 1:train_len]

        save_data['book_orders'] = dict()
        for coin, book_order in self._book_orders.items():
            save_data['book_orders'][coin] = book_order[train_len - self.look_back * self.stride - 1:train_len]

        f = open(f'models/last_data_{filename}.pkl', 'wb')
        pickle.dump(save_data, f, 2)
        f.close()

    def load_for_tarding(self, filename):
        self.load_scalers(filename)
        f = open(f'models/last_data_{filename}.pkl', 'rb')
        load_data = pickle.load(f)
        f.close()

        self.df = load_data['dataframe']
        self._book_orders = load_data['book_orders']

        self.df_scaled = self.df.copy()
        for coin, sc in self.scalers.items():
            x = self.df_scaled[coin].values
            x_scaled = sc.transform(x)
            self.df_scaled[coin] = x_scaled

        self.coins = list(self.scalers.keys())
        self.df.drop(columns=set(self.bad_coins).intersection(self.coins))
        for coin in self.bad_coins:
            if coin in self.coins:
                self.coins.remove(coin)
                self._book_orders.pop(coin, None)
                self.scalers.pop(coin, None)
        self.num_coins = len(self.coins)

    def copy_for_trading(self, dm):
        train_len = int(dm._train_test_ratio * dm.df.shape[0])
        self.scalers = dm.scalers
        self.df = dm.df[train_len - dm.look_back * dm.stride - 1:train_len].copy()

        self._book_orders = dict()
        for coin, book_order in dm._book_orders.items():
            self._book_orders[coin] = book_order[train_len - dm.look_back * dm.stride - 1:train_len]

        self.df_scaled = self.df.copy()
        for coin, sc in self.scalers.items():
            x = self.df_scaled[coin].values
            x_scaled = sc.transform(x)
            self.df_scaled[coin] = x_scaled

        self.coins = list(self.scalers.keys())
        self.df.drop(columns=set(self.bad_coins).intersection(self.coins))
        for coin in self.bad_coins:
            if coin in self.coins:
                self.coins.remove(coin)
                self._book_orders.pop(coin, None)
                self.scalers.pop(coin, None)
        self.num_coins = len(self.coins)

    def save_scalers(self, filename):
        f = open(f'models/scalers_{filename}.pkl', 'wb')
        pickle.dump(self.scalers, f, 2)
        f.close()

    def load_scalers(self, filename):
        f = open(f'models/scalers_{filename}.pkl', 'rb')
        scalers = pickle.load(f)
        f.close()

        self.scalers = scalers


if __name__ == '__main__':
    dm = DataManager(
        # price_or_buy_sell='price',
        # max_size_of_database=1000,
        # start_date=config.start_date_train,
        # end_date=config.end_date_train,
    )
    dm.initialize(config.db_path, )

    dm.plot_trends(plt.figure(figsize=(20, 10)).add_subplot(111))
    # dm.save('dm.pkl')
    plt.show()
