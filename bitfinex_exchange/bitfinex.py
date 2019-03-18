import hashlib
import hmac
import json
import time  # for nonce

import random
import requests


class BitfinexClient(object):
    BASE_URL = "https://api.bitfinex.com/"
    KEY = "sWHjR9dXHVjtvRakOpoB60x1p6iDb1WO6Vz3N2mw1B3"
    SECRET = "kiqwvg6iPBAyUCZHbbRPXwMBxuXky7E2Podt2qUTHvl"

    @staticmethod
    def _nonce():
        """
        Returns a nonce
        Used in authentication
        """
        return str(int(round(time.time() * 1000)))

    def _headers(self, path, nonce, body):

        signature = "/api/" + path + nonce + body
        print("Signing: " + signature)
        h = hmac.new(self.SECRET.encode('utf8'), signature.encode('utf8'), hashlib.sha384)
        signature = h.hexdigest()

        return {
            "bfx-nonce": nonce,
            "bfx-apikey": self.KEY,
            "bfx-signature": signature,
            "content-type": "application/json"
        }

    def wallets(self):
        nonce = self._nonce()
        body = {}
        raw_body = json.dumps(body)
        path = 'v2/auth/r/wallets'

        # print(self.BASE_URL + path)
        # print(nonce)

        headers = self._headers(path, nonce, raw_body)

        # print(headers)
        # print(raw_body)

        # print("requests.post(" + self.BASE_URL + path + ", headers=" + str(
        #     headers) + ", data=" + raw_body + ", verify=True)")
        r = requests.post(self.BASE_URL + path, headers=headers, data=raw_body, verify=True)

        if r.status_code == 200:
            return r.json()
        else:
            print(r.status_code)
            print(r)
            return ''

    def wallets_v1(self):
        nonce = self._nonce()
        body = {}
        raw_body = json.dumps(body)
        path = 'v1/balances'

        # print(self.BASE_URL + path)
        # print(nonce)

        headers = self._headers(path, nonce, raw_body)

        # print(headers)
        # print(raw_body)

        # print("requests.post(" + self.BASE_URL + path + ", headers=" + str(
        #     headers) + ", data=" + raw_body + ", verify=True)")
        r = requests.post(self.BASE_URL + path, headers=headers, data=raw_body, verify=True)

        if r.status_code == 200:
            return r.json()
        else:
            print(r.status_code)
            print(r)
            return ''

    def available(self, coin, buy=True):
        nonce = self._nonce()
        body = {'symbol': f't{coin}',
                'dir': 1 if buy else -1,  # buy
                'rate': 800,
                'type': 'EXCHANGE'
                }
        raw_body = json.dumps(body)
        path = 'v2/auth/calc/order/avail'

        # print(self.BASE_URL + path)
        # print(nonce)

        headers = self._headers(path, nonce, raw_body)

        # print(headers)
        # print(raw_body)

        # print("requests.post(" + self.BASE_URL + path + ", headers=" + str(
        #     headers) + ", data=" + raw_body + ", verify=True)")
        r = requests.post(self.BASE_URL + path, headers=headers, data=raw_body, verify=True)

        if r.status_code == 200:
            return r.json()
        else:
            print(r.status_code)
            print(r)
            return ''

    def new_order(self, coin, amount):
        nonce = self._nonce()
        body = {
            'cid': random.getrandbits(45),
            'type': 'exchange limit',
            'side': 'sell',
            'amount': amount,
            'price': 1,
            'symbol': f't{coin}',
        }
        raw_body = json.dumps(body)
        path = 'v1/order/new'

        # print(self.BASE_URL + path)
        # print(nonce)

        headers = self._headers(path, nonce, raw_body)

        # print(headers)
        # print(raw_body)

        # print("requests.post(" + self.BASE_URL + path + ", headers=" + str(
        #     headers) + ", data=" + raw_body + ", verify=True)")
        r = requests.post(self.BASE_URL + path, headers=headers, data=raw_body, verify=True)

        if r.status_code == 200:
            return r.json()
        else:
            print(r.status_code)
            print(r)
            return ''

    def active_orders(self):
        """
        Fetch active orders
        """
        nonce = self._nonce()
        body = {}
        raw_body = json.dumps(body)
        path = "v2/auth/r/orders"

        # print(self.BASE_URL + path)
        # print(nonce)

        headers = self._headers(path, nonce, raw_body)

        # print(headers)
        # print(raw_body)

        # print("requests.post(" + self.BASE_URL + path + ", headers=" + str(
        #     headers) + ", data=" + raw_body + ", verify=True)")
        r = requests.post(self.BASE_URL + path, headers=headers, data=raw_body, verify=True)

        if r.status_code == 200:
            return r.json()
        else:
            print(r.status_code)
            print(r)
            return ''


# bfx = BitfinexClient()
# print(bfx.active_orders())
# print(bfx.wallets())
# print(bfx.wallets_v1())
# print(bfx.available('MGOUSD', False))
# print(bfx.new_order('MGOUSD', 30))
