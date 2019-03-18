import datetime
import json

import numpy as np
import requests
import sys
import tables

urls = {
    'ticker': 'https://api.bitfinex.com/v2/ticker/t{}',
    'trades': 'https://api.bitfinex.com/v2/trades/t{}/hist',
    'book': 'https://api.bitfinex.com/v2/book/t{}/P1',
}


class TS(tables.IsDescription):
    timestamp = tables.Time64Col(pos=0)
    ticker = tables.Float64Col(shape=(10,))
    trades = tables.Float64Col(shape=(120, 4))
    book = tables.Float64Col(shape=(50, 3))


def get_symbols(_filter=True):
    if _filter:
        return [s.upper() for s in ['btcusd', 'ltcusd', 'ethusd',
                                    'etcusd', 'zecusd', 'xmrusd',
                                    'dshusd', 'xrpusd', 'iotusd',
                                    'neousd', 'eosusd', 'xlmusd',
                                    'trxusd', 'mgousd', 'bchusd']]
    # 'adausd', 'xemusd', 'vemusd', ]]
    else:
        sym_url = 'https://api.bitfinex.com/v1/symbols'
        response = requests.get(sym_url)
        return [s.upper() for s in json.loads(response.text)]


def get_book_orders(url_base, sym):
    try:
        with requests.get(url_base.format(sym), timeout=(2, 10)) as response:
            raw, status = response.text, response.status_code
        data = json.loads(raw)

    except requests.exceptions.ConnectTimeout:
        data = {
            'time': datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
            'error': 'timeout',
            'err': sys.exc_info()
        }
        print(data)
    except requests.exceptions.ConnectionError:
        data = {
            'time': datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
            'error': 'connection',
            'err': sys.exc_info()
        }
        print(data)

    if 'error' in data:
        raise Exception(
            text='{}, error, {}, {}'.format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'), sym, data))
    else:
        data = np.asarray(data)

    return sym, data
