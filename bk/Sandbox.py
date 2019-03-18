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

import Predictor_h5

from primal_trader import Exchanger
from first_trader import Exchanger

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

        models[db.name], rates[db.name], test_price[db.name] = Predictor_h5.learn(db, l, max_size=7100, mode='GRU')

prices = []
portfos = []

for currency in rates:
    prices.append(test_price[currency].ravel())
    portfos.append(rates[currency].ravel())

prices = np.array(prices)
portfos = np.array(portfos)

s = 0
for currency in rates:
    d = np.array(rates[currency].ravel())
    print('{0}: {1}'.format(currency, d))
    s += sum(d < 0)
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
