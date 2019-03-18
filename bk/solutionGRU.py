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

import numpy as np
import matplotlib.pyplot as plt
import tables

import Data5minPredictorGRU
import primal_trader

db_filename = './data/b.h5'
models = {}
rates = {}
test_price = {}

with tables.File(db_filename, 'r') as f:
    for l, db in enumerate(f.root):
        if l > 1:
            pass
            # continue
        if db[:]['ticker'].shape[0] < 7100:
            continue

        models[db.name], rates[db.name], test_price[db.name] = Data5minPredictorGRU.learn(db, l, max_size=7100)

prices = []
portfos = []

for k in rates:
    prices.append(test_price[k].ravel())
    portfos.append(rates[k].ravel())

prices = np.array(prices)
portfos = np.array(portfos)


class Exchanger:
    money = 100

    def __init__(self, prices, portfolio, log=False):
        assert prices.__len__() == portfolio.__len__()
        assert sum(portfolio) > 0

        self.prices = np.array(prices)
        portfolio = np.asarray(portfolio) / np.sum(portfolio)
        self.portfolio = np.zeros(len(portfolio))

        for i, x in enumerate(zip(prices, portfolio)):
            pr, x = x
            self.portfolio[i] = (self.money * x) / pr

        self.log = log
        if self.log:
            self.log_portfo = [self.portfolio]
            self.log_prices = [self.prices]

    def exchange(self, prices, portfolio):
        assert prices.__len__() == portfolio.__len__()

        portfolio[portfolio < 0] = 0

        if sum(self.portfolio > 0):
            self.money = 0
            for pr, x in zip(prices, self.portfolio):
                self.money += x * pr

        if sum(portfolio) == 0:
            self.portfolio = np.zeros(len(portfolio))
        else:

            # portfolio = np.asarray(portfolio) == np.max(portfolio)
            portfolio = np.asarray(portfolio) / np.sum(portfolio)
            print(portfolio)

            for i, x in enumerate(zip(prices, portfolio)):
                pr, x = x
                self.portfolio[i] = (self.money * x) / pr

            self.prices = prices

        if self.log:
            self.log_portfo.append(self.portfolio)
            self.log_prices.append(self.prices)


s = 0
for r in rates:
    d = np.array(rates[r].ravel())
    print('{0}: {1}'.format(r, d))
    s += sum(d<0)
print(s)

prs = prices[:, 0]
ports = portfos[:, 0]
ex = Exchanger(prs, ports, True)
trends = []

for i in range(1, prices.shape[1]):
    ex.exchange(prices[:, i], portfos[:, i])
    trends.append(ex.money)

plt.plot(trends)
plt.show()

for x in np.asarray(ex.log_prices).T:
    plt.semilogy(x)
plt.show()

for x in np.asarray(ex.log_portfo).T:
    plt.plot(x)
plt.show()

np.array(ex.log_portfo) > 0