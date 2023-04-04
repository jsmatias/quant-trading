import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import notebook
import json
import urllib.request
import re
from pathlib import Path
import re


class Broker:
    portfolio = {}
    metadata = {}
    closedTimes = [17, 10] # o'clock
    closedDays = ['sat', 'mon'] # day of week 
    _fees = '0.00 + 0.00%'
    _source = {}
    _calendarclock = None
    _openningTime = '09:00'
    _closingTime = '18:00'
    # _maxperiod = [datetime.today(), datetime(1900, 1, 1)]
    _host = 'https://www.alphavantage.co/query?'
    _apiKey = 'apikey=XXXXXXXXXXXXXXXX'
    _outfileDir = '../../source/database/'

    def __init__(self, trader) -> None:
        self._trader = trader
        with open('../../source/.key') as k:
            self._apiKey = k.readline()

    @staticmethod
    def _strtodate(datestr):
        """Converts a string date 'YYYY-MM-DD' to date format"""
        day, month, year = (int(n) for n in datestr.split('-'))
        dt = datetime(year, month, day)
        return(dt)

    @property
    def fees(self):
        return(self._fees)

    @fees.setter
    def fees(self, strExpression):
        self._fees = str(strExpression)
    
    @property
    def calendarclock(self):
        return(self._calendarclock)

    @calendarclock.setter
    def calendarclock(self, dateTime):
        if type(dateTime) is str:
            self._calendarclock = self._strtodate(dateTime)
        elif type(dateTime) is type(datetime.today()):
            self._calendarclock = dateTime

    def calculatefees(self, tradedCap):
        """Calculates the total fee for a trade order"""
        expArr = self.fees.split('+')
        vFixed = float(expArr[0]) if '%' not in expArr[0] else 0.0
        vVarible = float(expArr[0].strip()[:-1]) if '%' in expArr[0] else 0.0
        if len(expArr) == 2:
            vFixed = float(expArr[1]) if '%' not in expArr[1] else vFixed
            vVarible = float(expArr[1].strip()[:-1]) if '%' in expArr[1] else vVarible
        elif len(expArr) > 2:
            print('Inconsistency on the fees expression')
        fee = round(vFixed + vVarible * abs(tradedCap) / 100, 2)
        return(fee)

    def load(self, tickers=None, start=None, end=datetime.today(), timeseries='DAILY', download=False):

        tickers = set([st.ticker for st in self._trader.strategies]
                      if tickers is None else [tickers])
        pbar0 = notebook.trange(len(tickers), total=len(tickers), unit='ticker')
        for k in pbar0:
            ticker = list(tickers)[k]
            pbar0.set_description(desc=ticker)
            
            if re.match(r"^([0-9]+min)$", timeseries):
                query = f'function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={timeseries}&adjusted=True&outputsize=full&datatype=json'
                # print(query)
                # return
            else:
                outputzise = 'full' if end - start > timedelta(150) else 'compact'
                query = f'function=TIME_SERIES_{timeseries.upper()}&symbol={ticker}&outputsize={outputzise}&datatype=json'
            self.loaddata(query, download=download)
        pbar0.close()
        for strategy in self._trader.strategies:
            strategy.request()

    def candleidx(self, ticker, dateTime):
        """Finds the index of a candle for a ticker"""
        try:
            idx = self.portfolio[ticker].index.get_loc(dateTime)
        except KeyError:
            idx = -1
        return(idx)

    def triggered(self, ticker, price, when, side=None, orderType=None):
        candleIdx = self.candleidx(ticker, when)
        trigger = self.portfolio[ticker].iloc[candleIdx]['Low'] <= price <= self.portfolio[ticker].iloc[candleIdx]['High']

        if f'{side}-{orderType}' == 'sell-limit':
            if self.portfolio[ticker].iloc[candleIdx]['Open'] >= price:
                # sell at open
                return(round(self.portfolio[ticker].iloc[candleIdx]['Open'].item(), 2))
            elif trigger:
                # sell at price
                return(round(price, 2))
            else:
                return(None)
        elif f'{side}-{orderType}' == 'sell-stop':
            if self.portfolio[ticker].iloc[candleIdx]['Open'] <= price:
                # sell at open
                return(round(self.portfolio[ticker].iloc[candleIdx]['Open'].item(), 2))
            elif trigger:
                # sell at price
                spread = 0 # TODO: include spread
                return(round(price - spread, 2))
            else:
                return(None)
        elif f'{side}-{orderType}' == 'sell-market':
            # sell at price
            spread = 0 # TODO: include spread
            return(round(price - spread, 2))

        elif f'{side}-{orderType}' == 'buy-limit':
            if self.portfolio[ticker].iloc[candleIdx]['Open'] <= price:
                # buy at open
                return(round(self.portfolio[ticker].iloc[candleIdx]['Open'].item(), 2))
            elif trigger:
                # buy at price
                return(round(price, 2))
            else:
                return(None)
        elif f'{side}-{orderType}' == 'buy-stop':
            if self.portfolio[ticker].iloc[candleIdx]['Open'] >= price:
                # buy at open
                return(round(self.portfolio[ticker].iloc[candleIdx]['Open'].item(), 2))
            elif trigger:
                # buy at price
                spread = 0 # TODO: include spread
                return(round(price + spread, 2))
            else:
                return(None)
        elif f'{side}-{orderType}' == 'buy-market':
            # sell at price
            spread = 0 # TODO: include spread
            return(round(price + spread, 2))

    def loaddata(self, query, download=False, refresh=False):
        """Load data from query. Join query with host website"""
        symbolRegex = re.compile(r'symbol=[^&]+')
        ticker = symbolRegex.search(query).group().split('=')[1]

        if (ticker not in self.portfolio.keys()) or refresh:
            outfile = self._outfileDir + ticker + '.csv'
            if download or not(Path(outfile).is_file()):
                self._source[ticker] = "Source: alphavantage.com"
                # download
                hostQuery = f'{self._host}{query}&{self._apiKey}'
                # r = requests.get(hostQuery)
                # data = r.json()
                with urllib.request.urlopen(hostQuery) as url:
                    data = json.loads(url.read().decode())
                # convert to dataframe
                try:
                    self.metadata[ticker] = pd.Series(data['Meta Data'])
                    key = list(data.keys())
                    key.remove('Meta Data')
                    dataset = pd.DataFrame(data[key[0]]).transpose().sort_index(ascending=True)
                    dataset = dataset.astype(float)
                    dataset.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if re.match(r"^([0-9]+min)$", self._trader.backtest.timeframe.lower()):
                        dataset.index = pd.to_datetime(dataset.index, format='%Y-%m-%d %H:%M:%S')
                    else:
                        dataset.index = pd.to_datetime(dataset.index + ' 10:00', format='%Y-%m-%d %H:%M')
                    self.portfolio[ticker] = dataset

                    # save to disk
                    dataset.to_csv(outfile)
                    # with open(outfile, 'w') as outf:
                    #     json.dump(data, outf)
                except:
                    self.metadata[ticker], dataset = data["Error Message"], None
                    print(self.metadata)
                    print(f'... query: {hostQuery}')
                    
            else:
                self._source[ticker] = "Source: DB"
                dataset = pd.read_csv(outfile, index_col=0)
                dataset.index = pd.to_datetime(dataset.index, format='%Y-%m-%d %H:%M')
                self.portfolio[ticker] = dataset
                # with open(outfile) as f:
                #     data = json.load(f)

            
        else:
            print(f'{ticker} already loaded')

    def sma(self, period, at='Close'):
        label = f'sma{period}' if at == 'Close' else f'sma{period}-{at}'
        for ticker in self.portfolio.keys():
            self.portfolio[ticker][label] = self.portfolio[ticker][at].rolling(
                window=period).mean()

    def ema(self, period, at='Close'):
        label = f'ema{period}' if at == 'Close' else f'ema{period}-{at}'
        for ticker in self.portfolio.keys():
            self.portfolio[ticker][label] = self.portfolio[ticker][at].ewm(span=period).mean()

    def rsi(self, period, at='Close'):

        # RSI calculation
        for ticker in self.portfolio.keys():
            isGain = self.portfolio[ticker][at].diff(1) > 0
            isLoss = self.portfolio[ticker][at].diff(1) < 0
            self.portfolio[ticker]['diff'] = self.portfolio[ticker][at].diff(1)
            self.portfolio[ticker]['gain'] = self.portfolio[ticker].loc[isGain, 'diff']
            self.portfolio[ticker]['loss'] = self.portfolio[ticker].loc[isLoss, 'diff']
            self.portfolio[ticker]['gain'].fillna(0, inplace=True)
            self.portfolio[ticker]['loss'].fillna(0, inplace=True)

            avgLoss = (period - 1) * [np.nan] + [- self.portfolio[ticker]['loss'][:period].mean()]
            avgGain = (period - 1) * [np.nan] + [self.portfolio[ticker]['gain'][:period].mean()]
            for _, candle in self.portfolio[ticker].iloc[period:].iterrows():
                avgGain.append((avgGain[-1] * (period - 1) + candle['gain']) / period)
                avgLoss.append((avgLoss[-1] * (period - 1) - candle['loss']) / period)
                
            avgGain = np.array(avgGain)
            avgLoss = np.array(avgLoss)

            rsi = 100 - 100 / (1 + avgGain / avgLoss)
            self.portfolio[ticker][f'RSI{period}'] = rsi
            self.portfolio[ticker].drop(columns=['diff', 'gain', 'loss', ], inplace=True)

    def lasttradedday(self, ticker, daytime=None):
        """Returns the last traded day relative to the daytime.

        Args:
            ticker (str): Stock symbol to be searched on portfolio
            daytime (datetime): if empty it considers the current date from clock

        Return:
            datetime: index
        """
        daytime = daytime if daytime else self._clockDay
        return(self._portfolio[ticker].truncate(after=daytime).index[-1])
