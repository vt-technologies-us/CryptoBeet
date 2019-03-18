import numpy as np
import matplotlib.pyplot as plt
import tables


class Exchanger:
    money = 100

    def __init__(self, prices, portfolio, log=False):
        assert prices.__len__() == portfolio.__len__()
        assert sum(portfolio) > 0

        self.prices = np.array(prices)
        portfolio = np.asarray(portfolio) / np.sum(portfolio)
        self.portfolio = np.zeros(len(portfolio))
        self.portfolio_f = portfolio

        for i, z in enumerate(zip(prices, portfolio)):
            pr, x = z
            self.portfolio[i] = (self.money * x) / pr

        self.log = log
        if self.log:
            self.log_portfo = [self.portfolio_f]
            self.log_prices = [self.prices]

    def update_money(self, prices):
        assert self.prices.__len__() == prices.__len__()

        if sum(self.portfolio) > 0:
            self.money = 0
            for pr, x in zip(prices, self.portfolio):
                self.money += x * pr

        self.prices = prices

    def update_portfolio_fraction(self):
        for i, x in enumerate(zip(self.prices, self.portfolio)):
            pr, x = x
            self.portfolio_f[i] = pr * x / self.money

    def exchange(self, prices, portfolio):
        assert prices.__len__() == portfolio.__len__()

        self.update_money(prices)
        self.update_portfolio_fraction()

        portfolio = np.asarray(portfolio)
        portfolio[portfolio < 0] = 0

        if sum(portfolio) > 0:
            portfolio = np.asarray(portfolio) / np.sum(portfolio)
            dp = (self.portfolio_f - portfolio)

            p_sell = (dp > 0) * dp
            p_buy = (dp < 0) * dp

            if sum(p_buy) < 0:
                p_buy = p_buy / sum(p_buy)

                if sum(self.portfolio_f) > 0:
                    money = 0
                    for i, z in enumerate(zip(prices, p_sell)):
                        pr, x = z
                        if self.portfolio_f[i] > 0:
                            sell_f = 1 if x / self.portfolio_f[i] > 1 else x / self.portfolio_f[i]
                            money += sell_f * pr * self.portfolio[i] * 0.999
                            self.portfolio[i] *= (1 - sell_f)
                else:
                    money = self.money * 0.999

                for i, z in enumerate(zip(prices, p_buy)):
                    pr, x = z
                    self.portfolio[i] += (money * x) / pr * 0.999

        else:
            self.portfolio = portfolio

        if self.log:
            self.log_portfo.append(self.portfolio_f.copy())
            self.log_prices.append(self.prices.copy())


def main():
    prs = [1000, 100, 10, 1, 1000]
    pr = prs
    ports = [1] * len(prs)
    ex = Exchanger(prs, ports, True)

    trends = []
    for i in range(1000):
        dpr = np.random.randn(len(prs)) * 0.01
        pr = prs * (1 + dpr)

        prof = pr - ex.prices
        # prs = pr

        ex.exchange(pr, -dpr)
        trends.append(ex.money)

    plt.plot(trends)
    plt.show()

    for x in np.asarray(ex.log_prices).T:
        plt.semilogy(x)
    plt.show()

    for x in np.asarray(ex.log_portfo).T:
        plt.plot(x)
    plt.show()


if __name__ == '__main__':
    main()
