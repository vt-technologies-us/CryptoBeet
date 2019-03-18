import datetime


TRAIN_MODELS = True
db_ds = 'data/bitfinex_data.h5'
db_path = 'data/bout.h5'


# Training Parameters
feature_size = 2
start_date_train = datetime.datetime(2018, 10, 28)
end_date_train = datetime.datetime(2018, 11, 8)
train_test_ratio = .85
predictor_mode = 'DashGRU'
ensemble_learning = False
ensemble_learning_modes = ['ActGRU', 'TWSTDGRU', 'TWSTD2GRU', 'DashGRU', 'DashLSTMGRU', ]
stride = 1


# Trading Parameters
sequential_trading = True
sequential_trading_iterations = 5
start_date_sequential_trading = datetime.datetime(2018, 1, 20)
wallets = 100

bad_coins = [
    # 'BTCUSD',
    # 'DSHUSD',
    # 'ETHUSD',
    # 'EOSUSD',
    # 'ETCUSD',
    # 'ETHUSD',
    # 'IOTUSD',
    # 'NEOUSD',
    # 'LTCUSD',
    # 'XLMUSD',
    # 'XMRUSD',
    # 'XRPUSD',
    # 'TRXUSD',
    # 'ZECUSD',
]

max_in_portfolio = 1
stop_loss_ratio = 0.004


subplot_in_row = 1
subplot_in_col = 1


# Exchange Simulation Parameters
fee = 0.002
start_date_test = datetime.datetime(2018, 11, 8)
end_date_test = datetime.datetime.now()  # datetime.datetime(2018, 4, 16)
max_size_of_trade_set = 288  # np.inf

