import time
import json
import tables
import tstables
import requests
import numpy as np

from apscheduler.schedulers.blocking import BlockingScheduler


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
				    'neousd', 'eosusd', 
                                    'xlmusd', 'trxusd', ]]
                                    #'adausd', 'xemusd', 'vemusd', ]]
    else:
        sym_url = 'https://api.bitfinex.com/v1/symbols'
        response = requests.get(sym_url)
        return [s.upper() for s in json.loads(response.text)]


def crawl(filename='bitfinex_data.h5', symbols=get_symbols()):
    url = {'ticker': 'https://api.bitfinex.com/v2/ticker/t{}',
           'trades': 'https://api.bitfinex.com/v2/trades/t{}/hist',
           'book': 'https://api.bitfinex.com/v2/book/t{}/P1',
           }

    with tables.File(filename, 'a') as f:
        for sym in symbols[:]:
            if '/{}'.format(sym) not in f:
                tbl = f.create_table('/', sym, TS, 'Timestamp records: {}'.format(sym))
            else:
                tbl = getattr(f.root, sym)

            r = tbl.row

            for uk, uv in url.items():
                try:
                    response = requests.get(uv.format(sym), timeout=(2, 10))
                    data = json.loads(response.text)
                except requests.exceptions.ConnectTimeout as err:
                    data = {'error': 'timeout'}

                if 'error' in data:
                    print('error {}, {}'.format(sym, uk))
                    data = -1 * np.ones(r[uk].shape)
                else:
                    data = np.asarray(data)

                r[uk] = data

            r['timestamp'] = time.time()

            print(r)
            r.append()


if __name__ == '__main__':
    crawl()
    sched = BlockingScheduler({'apscheduler.job_defaults.max_instances': '3',
                               'apscheduler.timezone': 'UTC', })
    sched.add_job(crawl, 'interval', id='bitfinix_crawler', minutes=5)
    sched.start()

