import asyncio
import datetime
import json
import time

import aiohttp
import numpy as np
import tables

from apscheduler.schedulers.blocking import BlockingScheduler

from bitfinex_exchange.symbols import get_symbols, TS, urls
from standard import config


async def get_url(url):
    """Nothing to see here, carry on ..."""
    # sleepy_time = random.randint(2, 5)
    # await asyncio.sleep(sleepy_time)

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            # print(url, await response.text())
            return await response.text(), response.status
            # return response


async def get_data(uk, uv, sym, r):
    try:
        # response = await get_url(uv.format(sym))
        raw, status = await get_url(uv.format(sym))
        data = json.loads(raw)
        # response.close()
    except asyncio.TimeoutError as err:
        data = {
            'time': datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
            'error': 'timeout',
            'err': err.with_traceback()
        }
        print(data)
    except aiohttp.client_exceptions.ClientConnectionError as err:
        data = {
            'time': datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'),
            'error': 'connection',
            'err': err.with_traceback()
        }
        print(data)

    if 'error' in data:
        print('{}, error {}, {}, {}'.format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'), sym, uk, data))
        data = -1 * np.ones(r[uk].shape)
    else:
        data = np.asarray(data)

    return uk, data


async def get_coin(f, sym):
    r = getattr(f.root, sym).row

    futures = [get_data(uk, uv, sym, r) for uk, uv in urls.items()]
    # for i, future in enumerate(futures):  #
    for i, future in enumerate(asyncio.as_completed(futures)):
        uk, data = await future
        r[uk] = data

    return r


async def crawl(filename=config.db_ds, symbols=get_symbols()):
    with tables.File(filename, 'a') as f:
        for sym in symbols:
            if '/{}'.format(sym) not in f:
                f.create_table('/', sym, TS, 'Timestamp records: {}'.format(sym))

        futures_coin = [get_coin(f, sym) for sym in symbols]
        for i, future in enumerate(asyncio.as_completed(futures_coin)):
            r = await future
            r['timestamp'] = time.time()
            r.append()
            print(r)


def main():
    asyncio.run(crawl())


if __name__ == '__main__':
    main()
    sched = BlockingScheduler({'apscheduler.job_defaults.max_instances': '3',
                               'apscheduler.timezone': 'UTC', })
    sched.add_job(main, 'interval', id='bitfinix_crawler', minutes=5)
    sched.start()
