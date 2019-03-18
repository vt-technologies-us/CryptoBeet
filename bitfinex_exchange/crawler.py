import json
import time

import numpy as np
import requests
import tables
from apscheduler.schedulers.blocking import BlockingScheduler

from bitfinex_exchange.symbols import get_symbols, TS, urls
from standard import config


def crawl(filename=config.db_ds, symbols=get_symbols()):
    with tables.File(filename, 'a') as f:
        for sym in symbols[:]:
            if '/{}'.format(sym) not in f:
                tbl = f.create_table('/', sym, TS, 'Timestamp records: {}'.format(sym))
            else:
                tbl = getattr(f.root, sym)

            r = tbl.row

            for uk, uv in urls.items():
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


def main():
    crawl()


if __name__ == '__main__':
    main()
    sched = BlockingScheduler({'apscheduler.job_defaults.max_instances': '3',
                               'apscheduler.timezone': 'UTC', })
    sched.add_job(main, 'interval', id='bitfinix_crawler', minutes=5)
    sched.start()
