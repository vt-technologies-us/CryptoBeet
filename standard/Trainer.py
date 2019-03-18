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
import tabulate
from matplotlib import gridspec, pyplot as plt
from tensorflow.python.client import device_lib

from standard import config, DataManager, Predictor


class Trainer:
    def __init__(self, **kwargs):
        self.predictors = {}
        self.dm = DataManager.DataManager(start_date=config.start_date_train,
                                          end_date=config.end_date_train,
                                          train_test_ratio=config.train_test_ratio,
                                          max_size_of_database=100000,
                                          bad_coins=config.bad_coins)
        self.dm.initialize(config.db_path)

        self.figs = {'loss': [],
                     'test': [],
                     'time': [], }

        for coin in self.dm.coins:
            if config.ensemble_learning:
                self.predictors[coin] = Predictor.EnsemblePredictor(epochs=kwargs.get('epochs', 10),
                                                                    mode=config.ensemble_learning_modes)
            else:
                self.predictors[coin] = Predictor.Predictor(epochs=kwargs.get('epochs', 10),
                                                            mode=config.predictor_mode)
            self.predictors[coin].make_model()

        for fig_name in self.figs:
            for i in range(int(np.ceil(self.dm.num_coins / (config.subplot_in_row * config.subplot_in_col)))):
                self.figs[fig_name].append(plt.figure(figsize=(14, 7)))

    def learn(self):
        for coin in self.dm.coins:
            print(f'Start learning {coin}')
            train_data = self.dm.get_train(coin)
            self.predictors[coin].learn(train_data['x'], train_data['y'])

    def evaluate(self):
        accuracies = []
        for coin in self.dm.coins:
            train_data, test_data = self.dm.get_train(coin), self.dm.get_test(coin)
            acc_train = self.predictors[coin].evaluate(train_data['x'], train_data['y'])
            acc_test = self.predictors[coin].evaluate(test_data['x'], test_data['y'])
            accuracies.append([coin, acc_train, acc_test])

        res = tabulate.tabulate(accuracies, headers=['Accuracy', 'Train', 'Test'], tablefmt='psql', floatfmt='.2e')
        print(res)

    def plot_test(self):
        gs = gridspec.GridSpec(config.subplot_in_col, config.subplot_in_row)
        for i, coin in enumerate(self.dm.coins):
            x_in_plot = int(i / config.subplot_in_row) % config.subplot_in_col
            y_in_plot = i % config.subplot_in_row
            ax = self.figs['test'][int(i / (config.subplot_in_row * config.subplot_in_col))] \
                .add_subplot(gs[x_in_plot, y_in_plot])
            test_data = self.dm.get_test(coin)
            self.predictors[coin].plot_result(ax, test_data['x'], coin, test_data['t'])
            line = ax.plot(test_data['t'], test_data['y'], label='real data')
            plt.setp(line, linewidth=.5)
            ax.legend()

    def plot_train(self):
        gs = gridspec.GridSpec(config.subplot_in_col, config.subplot_in_row)
        for i, coin in enumerate(self.dm.coins):
            x_in_plot = int(i / config.subplot_in_row) % config.subplot_in_col
            y_in_plot = i % config.subplot_in_row
            ax = self.figs['test'][
                int(i / (config.subplot_in_row * config.subplot_in_col))
            ].add_subplot(gs[x_in_plot, y_in_plot])
            train_data = self.dm.get_train(coin)
            self.predictors[coin].plot_result(ax, train_data['x'], coin, train_data['t'])
            ax.plot(train_data['t'], train_data['y'], label='real data')
            ax.legend()

    def save_plots(self, image_format='svg'):
        for k, fig_list in self.figs.items():
            for i, fig in enumerate(fig_list):
                if len(fig.axes) > 0:
                    fig.savefig(f'images/{k}_{i}.{image_format}')

    def show_plots(self):
        for k, fig_list in self.figs.items():
            for i, fig in enumerate(fig_list):
                if len(fig.axes) > 0:
                    fig.show()

    def save_results(self):
        import scipy.io as sio
        res = {}
        for coin in self.dm.coins:
            x, y = self.dm.ds[coin]
            y_predict = self.predictors[coin].predict(x)
            res[coin] = y, y_predict

        sio.savemat('data/results.mat', res)

    def save(self):
        with open('models/coins.txt', 'w') as f:
            json.dump(self.dm.coins, f)
        for coin in self.dm.coins:
            self.predictors[coin].save(f'{coin}')
        self.dm.save_for_tarding(f'data_manager')

    def load(self):
        for coin in json.load(open('models/coins.txt')):
            if config.ensemble_learning:
                self.predictors[coin] = Predictor.EnsemblePredictor()
            else:
                self.predictors[coin] = Predictor.Predictor()
            self.predictors[coin].load(f'{coin}')


def main():
    print(device_lib.list_local_devices())


if __name__ == "__main__":
    main()
    t = Trainer()
    # t.load()
    t.learn()
    t.evaluate()
    t.plot_test()
    t.save_plots()
    t.save()
