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

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from standard import config, Predictor, SimulationExchange, DataManager, BitfinexExchange


class Trader:
    def __init__(self, **kwargs):
        self.fee = config.fee
        self.max_in_portfolio = config.max_in_portfolio

        self.max_money = 100
        self.last_money = 100

        self.stop_loss = kwargs.get('stop_loss', 0.07)
        self.start_loss = kwargs.get('start_loss', 0.06)
        self.restart_max_ratio = kwargs.get('stop_loss', 0.98)

        self.state = 'RUN'

    def initialize(self, predictors=None, data_manager=None, exchange='simulation'):
        if predictors:
            self.predictors = predictors
        else:
            self.predictors = {}
            self.load()

        wallets = {}
        for coin in self.predictors:
            wallets[coin] = 0
        wallets['USD'] = config.wallets

        if exchange == 'bitfinex_exchange':
            self.ex = BitfinexExchange.BitfinexExchange()
        else:
            self.ex = SimulationExchange.SimulationExchange(wallets, )

        if data_manager:
            self.dm = DataManager.DataManager()
            self.dm.copy_for_trading(data_manager)
        else:
            self.dm = DataManager.DataManager()
            self.dm.load_for_tarding('data_manager')

    def load(self):
        for coin in json.load(open('models/coins.txt')):
            if coin in config.bad_coins:
                continue
            print(f'Loading predictor for {coin}')
            if config.ensemble_learning:
                self.predictors[coin] = Predictor.EnsemblePredictor()
            else:
                self.predictors[coin] = Predictor.Predictor()

            self.predictors[coin].load(f'{coin}')

    def get_trades(self, predicts):
        # assert self.dm.shape[0] == hist_futures.shape[0]

        index_buy = int(iter(self.dm._book_orders.values()).__next__().shape[1] / 2)
        index_sell = 0

        book_orders = [book_order[-1] for book_order in self.dm._book_orders.values()]
        if 'USD' not in predicts:
            predicts['USD'] = np.asarray([[1, 1]])
            book_orders.append(book_orders[-1].copy())
            book_orders[-1][:index_buy] = np.array([1, np.inf, np.inf])
            book_orders[-1][index_buy:] = np.array([1, np.inf, -np.inf])

        coins = list(predicts.keys())
        usd_ind = coins.index('USD')

        predict_vector = np.asarray(list(predicts.values()))
        current_vector = np.asarray(book_orders)

        diff = predict_vector[:, index_sell, 0] - current_vector[:, index_buy, 0]
        rates = (diff / current_vector[:, index_sell, 0]).reshape((-1, 1))
        flows = rates.T - rates

        flows -= 2 * self.fee
        flows[:, usd_ind] += self.fee
        flows[usd_ind, :] += self.fee

        i_index, j_index = np.meshgrid(range(flows.shape[0]), range(flows.shape[0]))
        flows[i_index == j_index] = 0

        trade = {}
        for i, coin in enumerate(coins):
            sort_ind = flows[i].argsort()
            best_ind = sort_ind[~np.isnan(flows[i][sort_ind])][-1]
            if i != best_ind:
                trade[coin] = coins[best_ind]

        return trade

    def run(self):
        while self.ex.next():
            # print(f'{self.ex.time_index}: {self.ex.money}, {this_time:%Y-%m-%d %H:%M:%S}')
            if self.ex.money < 1:
                print(self.ex.wallets)

            this_time, book_orders = self.ex.get_book_orders_and_time()
            ds = self.dm.new_data(this_time, book_orders)
            predicts = {}
            for coin in self.dm.coins:
                predict = self.predictors[coin].predict(ds[coin][0])
                predicts[coin] = self.dm.scalers[coin].inverse_transform(predict)
            trades = self.get_trades(predicts)

            if set(np.unique(list(trades.values()))).intersection(trades.keys()):
                print('error found')

            aims_money = {aim: 0 for aim in np.unique(list(trades.values()))}

            # sell
            for coin, aim in trades.items():
                if coin != 'USD':
                    money = self.ex.sell_order(coin, self.ex.get_wallets()[coin])
                else:
                    money = self.ex.get_wallets()[coin]
                aims_money[aim] += money

            # buy
            for aim, money in aims_money.items():
                if aim != 'USD':
                    self.ex.buy_order(aim, money)

            if self.ex.time_index % 100 == 0:
                print(f'{self.ex.time_index}: {self.ex.money}, {this_time:%Y-%m-%d %H:%M:%S}')

    def plot_results(self, ax):
        self.dm.plot_trends(ax)
        ax.plot(self.ex.log_times, self.ex.log_money, label='Strategy')
        ax.legend(loc='upper left')

    def _plot_results(self):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        self.plot_results(ax)
        fig.savefig('images/trends.svg')
        fig.show()

    def plot_wallets_change(self, ax, show_on_money=False):
        df = pd.DataFrame(self.ex.log_wallets_fraction)
        if show_on_money:
            for c in df:
                df[c] *= self.ex.log_money
            df[df == 0] = np.nan
        ax.plot(df)
        ax.legend(df.columns, loc='upper left')

    def _plot_wallets_change(self, show_on_money=False):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        self.plot_wallets_change(ax, show_on_money)
        fig.savefig(f'images/{"wallets_on_trend" if show_on_money else "wallets"}.svg')
        fig.show()


if __name__ == "__main__":
    t = Trader()
    t.run()
