import difflib
import itertools
import json
import time
from multiprocessing.dummy import Pool as ThreadPool

import ccxt
import numpy as np
from ccxt.base import errors

from bitfinex_exchange.symbols import urls, get_book_orders, get_symbols
from standard import config, AbstractExchange


class BitfinexExchange(AbstractExchange.AbstractExchange):
    abbr_dic = np.array([
        ['BTCUSD', 'BTC/USDT', 'BTC'],
        ['LTCUSD', 'LTC/USDT', 'LTC'],
        ['ETHUSD', 'ETH/USDT', 'ETH'],
        ['ETCUSD', 'ETC/USDT', 'ETC'],
        ['ZECUSD', 'ZEC/USDT', 'ZEC'],
        ['XMRUSD', 'XMR/USDT', 'XMR'],
        ['DSHUSD', 'DASH/USDT', 'DASH'],
        ['XRPUSD', 'XRP/USDT', 'XRP'],
        ['IOTUSD', 'IOTA/USDT', 'IOTA'],
        ['NEOUSD', 'NEO/USDT', 'NEO'],
        ['EOSUSD', 'EOS/USDT', 'EOS'],
        ['XLMUSD', 'XLM/USDT', 'XLM'],
        ['TRXUSD', 'TRX/USDT', 'TRX'],
        ['MGOUSD', 'MGO/USDT', 'MGO'],
        ['BCHUSD', 'BCH/USDT', 'BCH'],
    ])

    def __init__(self):
        with open('bitfinex_exchange/bitfinex_secrets.json') as f:
            secrets = json.load(f)

        self.bfx = ccxt.bitfinex(
            {
                'apiKey': secrets['apiKey'],
                'secret': secrets['secret'],
                'verbose': True,  # switch it to False if you don't want the HTTP log
            })

        self._get_wallets()

        self._update_book_orders()

        super(BitfinexExchange, self).__init__()

    def _get_wallets(self):
        self._wallets = self.bfx.fetch_balance()

        self.wallets = self._to_old_format(self._wallets['free'])

    def get_wallets(self):
        if time.time() - self.time > 150:
            self._get_wallets()
        return self.wallets

    def _stop_order(self, coin, volume, price):
        symbol = difflib.get_close_matches(coin, self.bfx.symbols, 1)[0]
        order_type = 'stop'
        side = 'sell'
        order = self.bfx.create_order(symbol, order_type, side, volume, price * (1 - config.stop_loss_ratio))
        return order

    def buy_order(self, coin, money):
        money = min(self.wallets['USDT'], money)
        symbol = difflib.get_close_matches(coin, self.bfx.symbols, 1)[0]
        order_type = 'market'
        side = 'buy'
        # todo amount
        amount = self.volume_x_buy(money, self.get_status(coin))
        try:
            order = self.bfx.create_order(symbol, order_type, side, amount, )
            self._stop_order(coin, order['amount'], order['price'])
            return order['amount']

        except errors.InsufficientFunds:
            return 0

    def sell_order(self, coin, volume):
        # todo set volume
        volume = min(self.wallets[coin], volume)

        symbol = difflib.get_close_matches(coin, self.bfx.symbols, 1)[0]
        order_type = 'market'
        side = 'sell'

        try:
            order = self.bfx.create_order(symbol, order_type, side, volume, )
            # todo return money not amount
            return order['amount']

        except errors.InsufficientFunds:
            return 0

    def record_log(self):
        super(BitfinexExchange, self).record_log()
        self.log_times.append(time.time())

    def _update_money(self):
        self.money = 0
        for k, w in self.wallets.items():
            # assert w > 0
            if not np.all(self.get_status(k) == -1) and np.all(np.isfinite(self.get_status(k)[:, 0])) and w > 0:
                self.money += self.price_x_sell(w, self.get_status(k))[0]

        if self.money == 0:
            self.money = self.log_money[-1] if len(self.log_money) > 0 else 1e-6

        return self.money

    def _update_portfolio_fraction(self):
        wallets_fraction = self.wallets.copy()
        for k, w in self.wallets.items():
            if not np.all(self.get_status(k) == -1):
                wallets_fraction[k] = self.price_x_sell(w, self.get_status(k))[0] / self.money

        return wallets_fraction

    def _update_book_orders(self):
        self._book_orders = {}
        pool = ThreadPool(4)
        futures_coin = pool.starmap(get_book_orders, zip(itertools.repeat(urls['book']), get_symbols()))
        for sym, future in futures_coin:
            self._book_orders[sym] = future

    def next(self):
        #   cancel all open orders
        open_orders = self.bfx.fetch_open_orders()
        for order in open_orders:
            self.bfx.cancel_order(order['id'])

        #   get current wallets
        self._get_wallets()

        #   get current book orders
        self._update_book_orders()

        #   calculate money
        #   set logs
        self.record_log()

        res = self.time < time.time() - 300
        self.time = time.time()

        return res

    def get_status(self, coin: str):
        if coin == 'USD':
            return self._USD
        return self._book_orders[coin]

    def get_book_orders(self):
        book_orders = self._book_orders
        if 'USD' not in book_orders:
            book_orders['USD'] = self._USD
        return book_orders

    @staticmethod
    def _to_old_format(dic, coin_or_change_bar=True):
        api_column = 2 if coin_or_change_bar else 1
        if type(dic) == dict:
            res = {}
            for abbr in BitfinexExchange.abbr_dic:
                res[abbr[0]] = dic[abbr[api_column]] if abbr[api_column] in dic else 0.0
            return res

        if type(dic) == str:
            target_arr = BitfinexExchange.abbr_dic[:, 0]
            return target_arr[BitfinexExchange.abbr_dic[:, api_column] == dic][0]

    @staticmethod
    def _to_api_format(dic, coin_or_change_bar=True):
        api_column = 2 if coin_or_change_bar else 1
        if type(dic) == dict:
            res = {}
            for abbr in BitfinexExchange.abbr_dic:
                res[abbr[api_column]] = dic[abbr[0]] if abbr[0] in dic else 0.0
            return res

        if type(dic) == str:
            target_arr = BitfinexExchange.abbr_dic[:, api_column]
            return target_arr[BitfinexExchange.abbr_dic[:, 0] == dic][0]


def main():
    ex = BitfinexExchange()


if __name__ == '__main__':
    # main()
    ex = BitfinexExchange()
