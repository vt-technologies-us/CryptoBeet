import time

import numpy as np

from standard import config


class AbstractExchange:
    fee = config.fee
    _USD = np.array([[1, 1, np.inf], [1, 1, -np.inf]])

    @staticmethod
    def price_x_sell(volume, book_order):
        money = 0
        book_order = book_order[:int(book_order.shape[0] / 2)]
        if not np.any(np.isnan(book_order)):
            for h in book_order:
                dx = min(h[2], volume)
                money += dx * h[0]
                volume -= dx
                if volume == 0:
                    break

        return money, volume

    @staticmethod
    def price_x_buy(volume, book_order):
        money = 0
        book_order = book_order[int(book_order.shape[0] / 2):]
        if not np.any(np.isnan(book_order)):
            for h in book_order:
                dx = min(h[2], volume)
                money += dx * h[0]
                volume -= dx
                if volume == 0:
                    break
        # else:
        #     raise Exception('Market has not enough capacity')

        return money, volume

    @staticmethod
    def volume_x_buy(money, book_order):
        volume = 0
        book_order = book_order[int(book_order.shape[0] / 2):]
        if not np.any(np.isnan(book_order)):
            for h in book_order:
                dm = min(abs(h[2] * h[0]), money)
                volume += dm / h[0]
                money -= dm
                if money == 0:
                    break
        # else:
        #     raise Exception('Market has not enough capacity')

        return money, volume

    def __init__(self):
        self.money = 0
        if not hasattr(self, 'wallets'):
            self.wallets = {}

        self.log_times = []
        self.log_money = []
        self.log_wallets = []
        self.log_wallets_fraction = []

        self.time = time.time()

        self.record_log()

    def get_wallets(self):
        raise NotImplementedError

    def buy_order(self, coin, money):
        raise NotImplementedError

    def sell_order(self, coin, volume):
        raise NotImplementedError

    def record_log(self):
        self.log_wallets.append(self.wallets.copy())

        self._update_money()
        self.log_money.append(self.money)

        self.log_wallets_fraction.append(self._update_portfolio_fraction())

    def _update_money(self):
        raise NotImplementedError

    def _update_portfolio_fraction(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def get_status(self, coin: str):
        raise NotImplementedError

    def get_book_orders(self):
        raise NotImplementedError

    def get_book_orders_and_time(self):
        return NotImplementedError
