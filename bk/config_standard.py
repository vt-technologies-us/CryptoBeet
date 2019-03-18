import datetime, pytz

db_path = 'data/bitfinex.h5'

# Training Parameters
start_date_train = datetime.datetime(2018, 8, 1)
end_date_train = datetime.datetime.now()
train_test_ratio = 0.85

subplot_in_row = 2
subplot_in_col = 2

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

# Exchange Simulation Parameters
fee = 0.002
start_date_test = datetime.datetime(2018, 8, 19, 7, 59, 10, tzinfo=pytz.utc)
end_date_test = datetime.datetime.now()

# Trading parameter
max_in_portfolio = 1
