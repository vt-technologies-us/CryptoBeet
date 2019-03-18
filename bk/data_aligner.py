import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tables

from Good import config

f = tables.File(config.db_addr, 'r')

coins = list(f.root._v_children)
shapes = list(x.shape[0] for x in f.root)
max_len_coin = max((f.root._v_children), key=lambda x: f.root._f_get_child(x).shape[0])
max_len = getattr(f.root, max_len_coin).shape[0]
data = np.full((max_len, f.root._v_nchildren + 1), np.nan)
data[:, 0] = getattr(f.root, max_len_coin)[:]['timestamp']
indices = [0] * f.root._v_nchildren

for i in range(data.shape[0]):
    for k in range(len(indices)):
        j = indices[k]
        if abs(data[i, 0] - getattr(f.root, coins[k])[j]['timestamp']) < 150:
            data[i, k + 1] = getattr(f.root, coins[k])[j]['timestamp']
            indices[k] += 1

print(shapes, indices)

data[:, 0] = getattr(f.root, max_len_coin)[:]['timestamp']
indices = [0] * f.root._v_nchildren

for i in range(data.shape[0]):
    for k in range(len(indices)):
        j = indices[k]
        if abs(data[i, 0] - getattr(f.root, coins[k])[j]['timestamp']) < 150:
            data[i, k + 1] = getattr(f.root, coins[k])[j]['ticker'][6]
            indices[k] += 1

print(f'{shapes}\n{indices}')

plt.figure(figsize=(20, 15))
for i in range(1, data.shape[1]):
    plt.subplot(3, 4, i)
    plt.plot(data[:, i], '.')
    plt.xlim(0, max_len)
    plt.ylim(data[0, 0], data[-1, 0])
    plt.title((coins[i - 1], shapes[i - 1]))
plt.show()

df = pd.DataFrame(data[:, 1:], columns=coins)
print(df.corr())

print(df.diff().corr())
