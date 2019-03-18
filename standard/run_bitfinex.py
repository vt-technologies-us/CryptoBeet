import datetime
import os

from apscheduler.schedulers.blocking import BlockingScheduler

from standard import Trader

import sys


old_stdout = sys.stdout

def main():
    with open("message.log","a") as log_file:
        sys.stdout = log_file

        trd.run()
    
    sys.stdout = old_stdout

    with open('log', 'w') as f:
        print(trd.ex.log_money, file=f)
        print(trd.ex.log_times, file=f)
        print(trd.ex.log_wallets, file=f)
        print(trd.ex.log_wallets_fraction, file=f)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    trd = Trader.Trader()
    trd.initialize(exchange='bitfinex_exchange')
    main()

    print('job started')
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', id='main', minutes=5, max_instances=3,
                      end_date=datetime.datetime.now() + datetime.timedelta(days=1))
    scheduler.start()
