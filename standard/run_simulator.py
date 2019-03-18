import matplotlib.pyplot as plt
from standard import Trainer, Trader, config

if config.TRAIN_MODELS:  # 'load'
    trn = Trainer.Trainer(epochs=10)
    # trn.load()
    trn.learn()
    trn.evaluate()
    trn.plot_test()
    trn.save_plots()
    trn.show_plots()
    trn.save()

trd = Trader.Trader()
if config.TRAIN_MODELS:
    trd.initialize(trn.predictors, trn.dm)
else:
    trd.initialize()
trd.run()
trd._plot_results()
trd._plot_wallets_change()
trd._plot_wallets_change(True)
