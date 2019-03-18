import ccxt
import json

with open('bitfinex_secrets.json') as f:
    secrets = json.load(f)

ex = ccxt.bitfinex({
    'apiKey': secrets['apiKey'],
    'secret': secrets['secret'],
    'verbose': True,  # switch it to False if you don't want the HTTP log
})

if __name__ == "__main__":
    symbol = 'MGO/USDT'  # bitcoin contract according to https://github.com/ccxt/ccxt/wiki/Manual#symbols-and-market-ids
    order_type = 'ticker'
    side = 'sell'  # or 'buy'
    amount = 39.8
    price = 1.0  # or None

    # order = ex.create_order(symbol, order_type, side, amount, price)
    # print(order)

    bo = ex.fetch_order_book(symbol, params={'precision': 'P1'})
