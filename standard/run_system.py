import json
import matplotlib.pyplot as plt
import numpy as np
from standard import Trainer, Trader, config
import datetime

result_of_sequential_trading = []


def plot_sequential_results(trd):
    trd = np.ravel(trd)
    plt.plot(trd)
    plt.xlabel('trading history timelaps')
    plt.show()


if config.sequential_trading:

    for i in range(config.sequential_trading_iterations):

        config.start_date_train = config.start_date_train + datetime.timedelta(1)
        config.end_date_train = config.end_date_train + datetime.timedelta(1)
        config.start_date_test = config.start_date_test + datetime.timedelta(1)

        if config.TRAIN_MODELS:  # 'load'
            trn = Trainer.Trainer(epochs=8)
            # trn.load()
            trn.learn()
            trn.evaluate()
            trn.plot_test()
            trn.save_plots()
            trn.show_plots()
            trn.save()

        trd = Trader.Trader()
        if config.TRAIN_MODELS:  # 'load'
            trd.initialize(trn.predictors, trn.dm)
        else:
            trd.initialize()
        trd.run()
        trd._plot_results()
        trd._plot_wallets_change()
        trd._plot_wallets_change(True)
        result_of_sequential_trading.append(trd.ex.log_money)
        print(type(result_of_sequential_trading))
        config.wallets = result_of_sequential_trading[-1][-1]

    with open('sequential_trading/tradings.txt', 'a+') as trad:
        json.dump(result_of_sequential_trading, trad)
    plot_sequential_results(result_of_sequential_trading)
    # trad.plot_sequential_results(result_of_sequential_trading, trd.ex.log_money)

else:

    if config.TRAIN_MODELS:  # 'load'
        trn = Trainer.Trainer(epochs=2)
        # trn.load()
        trn.learn()
        trn.evaluate()
        trn.plot_test()
        trn.save_plots()
        trn.show_plots()
        trn.save()

    trd = Trader.Trader()
    if config.TRAIN_MODELS:  # 'load'
        trd.initialize(trn.predictors, trn.dm)
    else:
        trd.initialize()
    trd.run()
    trd._plot_results()
    trd._plot_wallets_change()
    trd._plot_wallets_change(True)

# def plot_sequential_results(trd, figr):
#     trd.dm.plot_trends(figr)
#     figr.plot(trd.ex.log_times, trd.ex.log_money, label='trade history')
#     figr.legend(loc='upper left')
#     fig = plt.figure(figsize=(20, 10))
#     ax = fig.add_subplot(111)
#     trd.plot_results(ax)
#     fig.savefig('images/sequential_trends.svg')
#     fig.xlable('the sequential trading plot')
#     fig.show()
