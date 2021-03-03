import datetime
import time
import warnings
from collections import Counter, defaultdict

import jqdatasdk
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from forex_python.converter import CurrencyRates
from jqdatasdk import api as jqdata
from plotly import graph_objects as go


class Strategy():

    def __init__(self, df_mcap=None, df_prices=None, df_volume=None, date=None):
        self.df_mcap = df_mcap
        self.df_prices = df_prices
        self.df_volume = df_volume
        self.date = date

    def convert_ticker(self, ticker=None):
        if 'XSHG' in ticker:
            ticker = ticker.replace('.XSHG', '.SH')
        elif 'XSHE' in ticker:
            ticker = ticker.replace('.XSHE', '.SZ')
        elif 'SH' in ticker:
            ticker = ticker.replace('.SH', '.XSHG')
        elif 'SZ' in ticker:
            ticker = ticker.replace('.SZ', '.XSHE')
        return ticker

    def next_trading_day(self):
        date = pd.to_datetime(self.date)
        open_days = self.df_mcap.index
        idx = open_days.get_loc(date, method='backfill')
        return open_days[idx]

    def business_days(self, date):
        date = pd.to_datetime(date)
        return self.df_mcap[str(date.year)+'-'+str(date.month)].index

    def get_earliest_date(self, stock=None):
        date = self.df_mcap[stock].first_valid_index()
        date = pd.to_datetime(date)
        return date

    def get_atvr(self, stock=None):
        date = pd.to_datetime(self.date)
        # make a list of all months in consideration a.k.a. last 12 months
        months = [date-pd.DateOffset(months=n) for n in range(1, 13)]
        # what if the stock has only been traded for less than 12 months?
        earliest = self.get_earliest_date(stock)
        if months[0] < earliest:
            m = int((date-earliest).days/30)
            months = [date-pd.DateOffset(months=n) for n in range(1, m)]

        # for each month, get trading values for each day
        mtvr = []
        for month in months:
            # get a list of all trading days in this month
            trading_days = self.business_days(month)
            # compute daily trading values for this month
            dtv = self.df_volume[stock].loc[trading_days[0]
                :trading_days[-1]].replace(0, np.nan).dropna()*1000
            # if there is missing data then skip this stock
            if len(dtv) == 0:
                return np.nan
            # get the median
            med = np.median(dtv)
            # get monthly traded value for this month
            mtv = med * len(dtv)
            # get the market cap at the end of this month
            last_day = dtv.index[-1]
            mkt_cap = df_mcap.at[last_day, stock]*10000
            # compute the monthly traded value ratio for each month
            mtvr.append(mtv/mkt_cap)
        # finally, compute the average traded value ratio
        atvr = np.mean(mtvr)*12
        return atvr

    def categorize_industries(self, stocks=[]):
        date = pd.to_datetime(self.date)
        stocks = [self.convert_ticker(stock) for stock in stocks]
        industries = jqdata.get_industry(stocks, date)
        d = defaultdict(list)
        # get industry of a stock and add to defultdict
        for stock in stocks:
            try:
                industry_name = industries[stock]['sw_l1']['industry_name']
                d[industry_name].append(self.convert_ticker(stock))
            except KeyError:
                continue
        return d

    def filter_eligibility(self):
        date = pd.to_datetime(self.date)
        # 1st Step: Exclude NaN values
        na_eligible = [
            stock for stock in self.df_mcap.columns if not np.isnan(self.df_mcap.at[date, stock])]
        print(
            f'\n[Screening...] Excluded {len(self.df_mcap.columns)-len(na_eligible)} companies with missing values.')

        # 2nd Step: Exclude companies with market cap not in $200M-1500M
        cap_eligible = []
        c = CurrencyRates()
        try:
            rate = c.get_rate('USD', 'CNY', date)
        except ConnectionError:
            time.sleep(10)
        min = rate*200000000
        max = rate*1500000000

        for stock in na_eligible:
            if min <= self.df_mcap.at[date, stock]*10000 <= max:
                cap_eligible.append(self.convert_ticker(stock))
        print(
            f'[Screening...] Excluded {len(na_eligible)-len(cap_eligible)} companies with market cap smaller than $200M or larger than $1500M.')

        # 3rd Step: Exclude companies labeled ST or ST*
        st_status = jqdata.get_extras(
            'is_st', cap_eligible, start_date=date, end_date=date)
        st_eligible = [
            self.convert_ticker(stock) for stock in st_status.columns if not st_status.at[date, stock]]
        print(
            f'[Screening...] Excluded {len(cap_eligible)-len(st_eligible)} companies labeled ST or ST* by regulators.')

        # 4th Step: Exclude companies with less than 6 months of trading history
        length_eligible = []
        for stock in st_eligible:
            start_date = self.get_earliest_date(stock)
            if date - start_date > datetime.timedelta(days=180):
                length_eligible.append(stock)
        print(
            f'[Screening...] Excluded {len(st_eligible)-len(length_eligible)} companies with less than 6 months of trading history.')

        # 5th Step: Exclude companies that fail liquidity screening
        liquidity_eligible = []
        dict_atvr = {}
        for stock in st_eligible:
            atvr = self.get_atvr(stock)
            dict_atvr[stock] = atvr
        atvr_values = list(dict_atvr.values())
        # drop NaN values
        atvr_values = [v for v in atvr_values if not np.isnan(v)]
        threshold = np.percentile(atvr_values, 20)
        # criteria 1: average traded value ratio >= 5%
        # criteria 2: belong to the top 80% of the ATVR values in universe
        for k, v in dict_atvr.items():
            if v >= 0.05 and v >= threshold:
                liquidity_eligible.append(k)
        print(
            f'[Screening...] Excluded {len(length_eligible)-len(liquidity_eligible)} companies with inadequate liquidity.')

        # 6th Step: Adjust target market representation
        representation_eligible = []
        industry_dict = self.categorize_industries(liquidity_eligible)
        for industry in industry_dict.keys():
            # rank stocks within industry by descending ATVR
            industry_stocks = [
                stock for stock in industry_dict[industry] if not np.isnan(dict_atvr[stock])]
            industry_stocks = sorted(
                industry_stocks, key=lambda x: self.get_atvr(x), reverse=True)
            # full market cap
            industry_mkt_cap = sum([df_mcap.at[self.date, stock]
                                    for stock in na_eligible]) * 10000
            # the cut-off market representation is 40%
            industry_threshold = industry_mkt_cap * 0.4
            # select stocks until market_cap reaches threshold
            industry_eligible = []
            # if there is only one stock within the industry, this stock is included
            if len(industry_dict[industry]) == 1:
                industry_eligible.append(industry_dict[industry][0])
            else:
                current_cap = 0
                idx = 0
                while current_cap <= industry_threshold:
                    current_cap += self.df_mcap.at[date,
                                                   industry_stocks[idx]] * 10000
                    idx += 1
                    try:
                        industry_eligible.append(industry_stocks[idx])
                    except IndexError:
                        break

            representation_eligible += industry_eligible
        print(
            f'[Screening...] Excluded {len(liquidity_eligible)-len(representation_eligible)} companies for representativeness.')

        return representation_eligible


class Portfolio():

    def __init__(self, stocks=None, cash=0, df_prices=None):
        self.stocks = stocks
        self.cash = cash
        self.df_prices = df_prices

    def get_stock_liquidation(self, date=None):
        agg_stock_value = 0
        sell = []

        for stock, shares in self.stocks.items():
            price = self.df_prices.at[date, stock]
            stock_value = price * shares
            # was this stock suspended for trading?
            if np.isnan(self.df_prices.at[date, stock]):
                # when was this stock last traded?
                last_traded = self.df_prices[stock].last_valid_index()
                self.cash += df_prices.at[last_traded, stock]
                sell.append(stock)
            else:
                agg_stock_value += stock_value

        for stock in sell:
            del self.stocks[stock]

        return agg_stock_value

    def get_net_liquidation(self, date=None):
        net_liquidation = self.get_stock_liquidation(date) + self.cash
        return net_liquidation


class BackTest():

    def __init__(self, start_date='2010-01-01', end_date='2020-12-31', init_funds=10000000, commission=0, log=[], data=None):
        self.start_date = start_date
        self.end_date = end_date
        self.init_funds = init_funds
        self.commission = commission
        self.log = log
        self.data = data

    def next_trading_day(self, date=None):
        date = pd.to_datetime(date)
        open_days = self.data[0].index
        idx = open_days.get_loc(date, method='backfill')
        return open_days[idx]

    def get_portfolio(self, funds_available=None, cash_ratio=0.1, date=None, weight='cap'):
        strategy = Strategy(self.data[0], self.data[1], self.data[2], date)
        composition = strategy.filter_eligibility()
        df_mcap, df_prices = self.data[0], self.data[1]
        funds_investable = funds_available*(1-cash_ratio)
        cash = funds_available*cash_ratio
        portfolio_stocks = {}

        # stocks in portfolio are weighted based on market cap
        if weight == 'cap':
            agg_market_cap = sum([df_mcap.at[date, stock]
                                  for stock in composition])

            for stock in composition:
                try:
                    # the fund allocated to a single stock
                    ratio = df_mcap.at[date, stock] / agg_market_cap
                    amount = funds_investable * ratio
                    # calculate how many shares of stock can be bought
                    price = df_prices.at[date, stock]
                    shares = int(amount/price)

                    portfolio_stocks[stock] = shares
                    cash += (amount - price*shares)
                except ValueError:
                    continue

        # stocks in portfolio are equally weighted
        elif weight == 'equal':
            n = len(composition)
            # amount available for a single stock in portfolio
            amount = funds_investable / n
            # calculate how many shares of stock can be bought
            for stock in composition:
                price = df_prices.at[date, stock]
                shares = int(amount / price)
                portfolio_stocks[stock] = shares
                cash += (amount - price*shares)

        portfolio = Portfolio(stocks=portfolio_stocks,
                              cash=cash, df_prices=df_prices)
        return portfolio

    def calculate_pl(self, date=None):
        i = 0
        flag = 0
        # at the current date, which portfolio am I holding?
        for i in range(len(self.log)-1):
            if self.log[i][0] <= date < self.log[i+1][0]:
                flag = i
            else:
                i += 1
        current_portfolio = self.log[flag][1]
        # how much is my holding portfolio's worth?
        net_liquidation = current_portfolio.get_net_liquidation(date)
        # calculate profits & losses
        pl = (net_liquidation/self.init_funds) * 100

        return pl

    def run_backtest(self, freq=6):
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        now = self.next_trading_day(start_date)

        # intitial portfolio
        p = self.get_portfolio(
            funds_available=self.init_funds, date=now, weight='cap')
        self.log.append((now, p))

        while now <= end_date-pd.DateOffset(months=6):
            # next date for reblancing portfolio
            now += relativedelta(months=freq)
            now = self.next_trading_day(now)
            print(f'\n[Rebalancing...] Time: {now.date()}')
            # how much is the portfolio is worth now
            net_liquidation = p.get_net_liquidation(date=now)
            # make a new portfolio
            p = self.get_portfolio(funds_available=net_liquidation, date=now)

            self.log.append((now, p))

    def plot_performance(self):
        df_performance = jqdata.get_price(
            '000300.XSHG', start_date=self.start_date, end_date=self.end_date)
        df_performance = df_performance[['close']].rename(
            columns={'close': 'CSI 300'})
        # convert to percentage
        df_performance = df_performance / df_performance['CSI 300'][0]*100
        # get performance of portfolio
        dates = df_performance.index
        portfolio_performance = [self.calculate_pl(date) for date in dates]
        df_performance['Small Cap'] = portfolio_performance

        # plot graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_performance.index,
                                 y=df_performance['CSI 300'], mode='lines', name='CSI 300'))
        fig.add_trace(go.Scatter(x=df_performance.index,
                                 y=df_performance['Small Cap'], mode='lines', name='Small Cap'))
        fig.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # log into account
    jqdatasdk.auth('18070536824', '536824')
    queries = jqdata.get_query_count(field='spare')
    print(f'[Initilizing...] {queries} queries left today.')

    # load in datasets
    print('[Initilizing...] Loading data.')
    df_mcap = pd.read_csv('market_cap.csv', parse_dates=[
                          'date']).set_index('date')
    df_prices = pd.read_csv('price.csv', parse_dates=[
                            'date']).set_index('date')
    df_volume = pd.read_csv('volume.csv', parse_dates=[
                            'date']).set_index('date')
    print('[Initilizing...] Data successfully loaded.\n')

    # run backtest
    bt = BackTest(start_date='2010-01-01', end_date='2020-12-31',
                  data=(df_mcap, df_prices, df_volume))
    bt.run_backtest()
    bt.plot_performance()
