import datetime
import itertools
import json
import time
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import requests
import tables
from apscheduler.schedulers.blocking import BlockingScheduler

from bitfinex_exchange.symbols import get_symbols, TS, urls
from standard import config


def get_url(url):
    with requests.get(url, timeout=(2, 10)) as response:
        return response.text, response.status_code


def get_data(uk, uv, sym, r):
    try:
        # response = get_url(uv.format(sym))
        raw, status = get_url(uv.format(sym))
        data = json.loads(raw)
        # response.close()

    except requests.exceptions.ConnectTimeout as err:
        data = {
            'time': datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
            'error': 'timeout',
            'err': err.with_traceback()
        }
        print(data)

    if 'error' in data:
        print('{}, error {}, {}, {}'.format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'), sym, uk, data))
        data = -1 * np.ones(r[uk].shape)
    else:
        data = np.asarray(data)

    return uk, data


def get_coin(f, sym):
    r = getattr(f.root, sym).row

    # futures = [get_data(uk, uv, sym, r) for uk, uv in urls.items()]
    pool = ThreadPool(4)
    futures = pool.starmap(get_data, zip(urls.keys(), urls.values(), itertools.repeat(sym), itertools.repeat(r)))

    # for i, future in enumerate(futures):  #
    for i, future in enumerate(futures):
        uk, data = future
        r[uk] = data

    return r


def crawl(filename=config.db_ds, symbols=get_symbols()):
    with tables.File(filename, 'a') as f:
        for sym in symbols:
            if '/{}'.format(sym) not in f:
                f.create_table('/', sym, TS, 'Timestamp records: {}'.format(sym))

        # futures_coin = [get_coin(f, sym) for sym in symbols]
        pool = ThreadPool(4)
        futures_coin = pool.starmap(get_coin, zip(itertools.repeat(f), symbols))

        for i, future in enumerate(futures_coin):
            r = future
            r['timestamp'] = time.time()
            r.append()
            print(r)


def main():
    crawl()


if __name__ == '__main__':
    main()
    sched = BlockingScheduler({'apscheduler.job_defaults.max_instances': '3',
                               'apscheduler.timezone': 'UTC', })
    sched.add_job(main, 'interval', id='bitfinix_crawler', minutes=5)
    sched.start()
