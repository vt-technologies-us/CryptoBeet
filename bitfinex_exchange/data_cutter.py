import datetime

import os
import tables

from standard import config
from bitfinex_exchange.symbols import TS


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    with tables.open_file(config.db_ds) as f:
        with tables.File(config.db_path, 'w') as f_out:
            start_date = config.start_date_train
            end_date = datetime.datetime.now()

            for sym in f.root:
                if '/{}'.format(sym.name) not in f_out:
                    f_out.create_table('/', sym.name, TS, 'Timestamp records: {}'.format(sym))

                period = sym.get_where_list(
                    '(timestamp > {}) & (timestamp < {})'.format(start_date.timestamp(), end_date.timestamp()))

                getattr(f_out.root, sym.name).append(sym[period])


if __name__ == "__main__":
    main()
