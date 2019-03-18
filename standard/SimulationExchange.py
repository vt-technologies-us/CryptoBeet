import numpy as np

from standard import config, DataManager, AbstractExchange


class SimulationExchange(AbstractExchange.AbstractExchange):
    def __init__(self, wallets: dict):
        if 'USD' not in wallets:
            wallets['USD'] = 0

        self.dm = DataManager.DataManager(start_date=config.start_date_test,
                                          end_date=config.end_date_test,
                                          train_test_ratio=0,
                                          max_size_of_database=config.max_size_of_trade_set,
                                          bad_coins=config.bad_coins)
        # self.dm.initialize(config.db_path)
        self.dm._load_databases(config.db_path)

        self.wallets = wallets

        self.time_index = 0
        for coin in set(self.wallets.keys()).difference(self.dm.df.columns.levels[0]).difference(['USD']):
            self.wallets.pop(coin)

        super(SimulationExchange, self).__init__()

    def get_wallets(self):
        return self.wallets

    def buy_order(self, coin, money):
        money = min(self.wallets['USD'], money)

        if not np.all(self.get_status(coin) == -1) and money > 0:
            money, volume = self.volume_x_buy(money, self.get_status(coin))
            self.wallets['USD'] = money
            self.wallets[coin] += volume
            return volume
        else:
            return 0

    def sell_order(self, coin, volume):
        volume = min(self.wallets[coin], volume)

        if not np.all(self.get_status(coin) == -1) and volume > 0:
            money, volume = self.price_x_sell(volume, self.get_status(coin))
            self.wallets[coin] = volume
            self.wallets['USD'] += money
            return money
        else:
            return 0

    def record_log(self):
        super(SimulationExchange, self).record_log()
        self.log_times.append(self.dm.df.index[self.time_index])

    def _update_money(self):
        self.money = 0
        for k, w in self.wallets.items():
            # assert w > 0
            if not np.all(self.get_status(k) == -1) and np.all(np.isfinite(self.get_status(k)[:, 0])) and w > 0:
                self.money += self.price_x_sell(w, self.get_status(k))[0]

        if self.money == 0:
            self.money = self.log_money[-1]

        return self.money

    def _update_portfolio_fraction(self):
        wallets_fraction = self.wallets.copy()
        for k, w in self.wallets.items():
            if not np.all(self.get_status(k) == -1):
                wallets_fraction[k] = self.price_x_sell(w, self.get_status(k))[0] / self.money

        return wallets_fraction

    def next(self):
        self.record_log()
        self.time_index += 1

        return self.time_index < iter(self.dm._book_orders.values()).__next__().shape[0]

    def get_status(self, coin: str):
        if coin == 'USD':
            return self._USD
        return self.dm._book_orders[coin][self.time_index]

    def get_book_orders(self):
        book_orders = self.dm.get_book_orders(self.time_index)
        if 'USD' not in book_orders:
            book_orders['USD'] = self._USD
        return book_orders

    def get_book_orders_and_time(self):
        return self.dm.df.index[self.time_index], self.get_book_orders()


def main():
    ex = SimulationExchange({})


if __name__ == '__main__1':
    main()
