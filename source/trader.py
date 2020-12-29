import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import pandas_datareader as pdr
from datetime import datetime, timedelta
import plotly.graph_objects as go


class Trader():
    _portfolio = {}
    _period = []
    _R = 100
    _RRatio = 100 / 10000
    _trg = 200
    _initial_cap = 10000
    _capital = 0
    _fees = 0
    _clockDay = datetime.today()
    _highNLows = False
    _benchmark = pd.DataFrame(columns=[
        'entry_date',
        'pnl'
    ])
    _history = pd.DataFrame(columns=[
        'entry_date',
        'ticker',
        'entry',
        'entry_vol',
        'stop',
        'target',
        'status',
        'exit',
        'exit_vol',
        'pnl',
        'exit_date',
        'time_in'
    ])
    _summary = pd.Series(index=[
        'period',
        'avg_time_in',
        'total_trades',
        'open_trades',
        'pnl',
        'pnl_percent',
        'batting_avg',
        'sharpe',
        'liquid_cap',
        'total_cap',
    ], dtype='float64', name='summary')

    def __init__(self, risk_size, target_size, capital, fees):
        self._initialSettings = {
            'risk_size': risk_size,
            'target_size': target_size,
            'capital': capital,
            'fees': fees
        }
        self.reset()

    
    @staticmethod
    def _risk(entry, stop, vol):
        return(
            vol * (entry - stop)
        )

    @staticmethod
    def _volume(entry, stop, risk):
        return(
            round(risk / (entry - stop))
        )

    def reset(self):
        """
        """
        self._highNLows = False
        self._capital = self._initialSettings['capital']
        self._initial_cap = self._initialSettings['capital']
        self._R = self._initialSettings['risk_size']
        self._trg = self._initialSettings['target_size']
        self._fees = self._initialSettings['fees']
        self._RRatio = self._R / self._capital
        self._history = pd.DataFrame(columns=[
            'entry_date',
            'ticker',
            'entry',
            'entry_vol',
            'stop',
            'target',
            'status',
            'exit',
            'exit_vol',
            'pnl',
            'exit_date',
            'time_in'
        ])

    def updaterisk(self, every=50):
        """Updates the risk based on the capital and initial risk ratio.
        The update will be applied everytime the capital reaches initial capital * (1 + every / 100).

        Args:
            every (float): every in percent of initial cap
        """

        currentRatio = self._R / self.totalcap()
        rr = self._trg / self._R

        if currentRatio < (self._RRatio / (1 + every / 100)):
            # update
            self._R = round(self._RRatio * self.totalcap(), 0)
            self._trg = round(rr * self._R)
            print(currentRatio, self._RRatio)
            print(f'Risk updated: {self._R}')
            print(f'Trg updated: {self._trg}')

    def trail(self, ticker, raiseToRisk=0, whenGain=2):
        """Raises the stop to another level when the price reaches certain gain.

        Args:
            ticker (str): ticker symbol for which the stop loss will be raised
            raiseToRisk (float): value proportional to R larger than -1
            whenGain (float): value proportional to R
        """
        if raiseToRisk > -1:
            boughtFilter = (self._history['status'] == 'bought') & (self._history['ticker'] == ticker)
            for idx, position in self._history[boughtFilter].iterrows():
                riskSize = position['entry'] - position['stop']
                if (self._portfolio[ticker].loc[self.lasttradedday(ticker), 'High'] >= position['entry'] + whenGain * riskSize):
                    # raise stop
                    self._history.loc[idx, 'stop'] = position['entry'] + raiseToRisk * riskSize

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

    def loadportfolio(self, portfolio, start=None, end=None):
        """
        Args:
            portfolio (list): list of tickers, e.g. MGLU3.SA, TSLA
            start (datetime): If None, takes 5 years ago
            end (datetime):
        """
        firstDay = datetime.today()
        lastDay = datetime(1900, 1, 1)
        self._portfolio = {}
        for ticker in portfolio:
            self._portfolio[ticker] = pdr.get_data_yahoo(
                ticker,
                start=start,
                end=end
            )
            firstDay = self._portfolio[ticker].index[0] if self._portfolio[ticker].index[0] < firstDay else firstDay
            lastDay = self._portfolio[ticker].index[-1] if self._portfolio[ticker].index[-1] > lastDay else lastDay

        self._period = [firstDay, lastDay]

    def testperiod(self):
        return(self._period)
        
    def portfolio(self, ticker=None):
        """Returns list of traded symbols
        """
        if ticker:
            return(self._portfolio[ticker])
        else:
            return(list(self._portfolio.keys()))

    def settarget(self, pos_idx, target=0):

        if pos_idx >= 0:
            entry, stop = self._history.loc[pos_idx, ['entry', 'stop']]
            target = target if target else (self._trg / self._R) * (entry - stop) + entry
            self._history.loc[pos_idx, 'target'] = target
        else: print("Couldn't find the trade to set target to...")

    def buy(self, ticker, entry, stop, entry_date, vol=0):
        vol = vol if vol else self._volume(entry, stop, self._R)

        if vol * entry < self._capital:

            # target = (self._trg / self._R) * (entry - stop) + entry
            self._history = self._history.append(
                pd.DataFrame({
                    'entry_date': [entry_date],
                    'ticker': [ticker],
                    'entry': [entry],
                    'entry_vol': [vol],
                    'stop': [stop],
                    # 'target': [target],
                    'status': ['bought']
                })
            ).reset_index(drop=True)
            self._capital -= entry * vol

            return(self._history.index[-1])

        else: 
            print('Low capital...')
            return(-1)

    def sell(self, price, exit_date, pos_idx, vol=0):

        # vol else sell all
        vol = vol if vol else self._history.loc[pos_idx, 'entry_vol']
        ticker = self._history.loc[pos_idx, 'ticker']
        # print(vol, self.totalvol())
        if self.totalvol()[ticker] >= vol:
            pnl = vol * (price - self._history.loc[pos_idx, 'entry'])
            self._history.loc[pos_idx, 'exit_vol'] = -vol
            self._history.loc[pos_idx, 'exit_date'] = pd.to_datetime(exit_date)
            self._history.loc[pos_idx, 'exit'] = price
            self._history.loc[pos_idx, 'pnl'] = pnl - self._fees
            self._history.loc[pos_idx, 'status'] = 'sold'
            time_in = self._history.loc[pos_idx, 'exit_date'] - self._history.loc[pos_idx, 'entry_date']
            self._history.loc[pos_idx, 'time_in'] = f'{time_in.days} days'

            self._capital += vol * price - self._fees
            return(pos_idx)
        else:
            print('Not enough volume...')
            return(-1)

    #  def addreduce(self, entry, stop):
        # 'returns new entry volume'
        # self._entry.append(entry)
        # self._stop.append(stop)

        # actualRisk = self._R - sum([self._risk(e, stop, v) for e, v in zip(self._entry, self._vol)])
        # vol = self._volume(entry, stop, actualRisk)
        # self._vol.append(vol)

        # print(f'New vol: {vol}')
     
    def backtest(self, entrymodel, exitmodel, updaterisk=False, trail=False):
        """Simulates trades over the chosen period

        Args:
            model (class): It must have the following prototype:
                class model:
                    def callback(trader:Trader, ticker, dayIdx):    
                        # if entry opportunity?: 
                            trader.buy()
                            # entry strategy() (buy or add)
                
                    # exit?:
                        # exit strategy() (trader.sell())
        """
        if self._portfolio:
            entryObjPerTicker = {ticker: entrymodel(ticker, self) for ticker in self._portfolio}
            exitObjPerTicker = {ticker: exitmodel(ticker, self) for ticker in self._portfolio}

            self._clockDay = self.testperiod()[0] + timedelta(1)
            while self._clockDay < self.testperiod()[1]:
                for ticker in self._portfolio:
                    try:
                        dayIdx = self._portfolio[ticker].index.get_loc(self._clockDay)
                    except KeyError:
                        dayIdx = 0
                    if dayIdx > 0:
                        # entry?
                        entryObjPerTicker[ticker].entrycallback(dayIdx) # callback method
                        # exit?
                        exitObjPerTicker[ticker].exitcallback(dayIdx) # callback method
                        
                        if updaterisk:
                            self.updaterisk(every=50)

                        if trail:
                            self.trail(ticker)

                self._clockDay += timedelta(1)
        else:
            print('Create portfolio before backtesting...')
    
    def history(self, ticker=''):
        if ticker:
            mask = self._history['ticker'] == ticker
            return(self._history[mask])
        else:
            return(self._history)
    
    def summary(self, ticker=''):
        """Returns results over the backtested period.
        """
        self._summary['period'] = self._period
        self._summary['avg_time_in'] = self.history(ticker).loc[self.history(ticker)['status']=='sold', 'time_in'] \
            .apply(lambda s: int(s.split(' ')[0])).mean()
        self._summary['total_trades'] = self.history(ticker).shape[0]
        self._summary['open_trades'] = self.history(ticker)[self.history(ticker)['status']=='bought'].shape[0]
        self._summary['pnl'] = self.pnl(ticker)
        self._summary['pnl_percent'] = self.pnl(ticker) * 100 / self._initial_cap
        self._summary['batting_avg'] = self.battingavg(ticker)
        self._summary['sharpe'] = self.sharpe(ticker)
        self._summary['liquid_cap'] = self.capital()
        self._summary['total_cap'] = self.totalcap()

        idxs = self._summary.index[:-2] if ticker else self._summary.index
        return(self._summary[idxs])

    def capital(self):
        return(self._capital)

    def totalcap(self, atDate=None):
        """Calculates total capital on the date: atDate.
            totalcap = capital + sum(volume_ticker * price_atDate)

        Args:
            atDate (datetime | int): if -1 date is selected as last day with data available for the current ticker
        
        Return:
            cap (float): total capital calculated atDate.
        """
        atDate = atDate if atDate else self._clockDay
        volDf = self.totalvol()
        cap = self.capital()
        for ticker in volDf.index:
            lastTradedCandle = self._portfolio[ticker].loc[self.lasttradedday(ticker, daytime=atDate)]
            priceAtDate = self._portfolio[ticker].iloc[-1]['Close'] if atDate==-1 else lastTradedCandle['Close']
            cap += volDf[ticker] * priceAtDate
        return(cap)
    
    def totalvol(self):
        """Calculates total volume of open trades        
        """

        groupByTicker = self._history.groupby('ticker')
        return(
            groupByTicker['entry_vol'].sum() + groupByTicker['exit_vol'].sum()
        )
        
    def pnl(self, ticker=''):
        return(
            self.history(ticker)['pnl'].sum()
        )
    
    def battingavg(self, ticker=''):
        return(
            self.history(ticker)[self.history(ticker)['pnl'] > 0].shape[0] / self.history(ticker)[self.history(ticker)['status'] == 'sold'].shape[0]
        )
    
    def sharpe(self, ticker=''):
        avgWin = self.history(ticker).loc[self.history(ticker)['pnl'] > 0, 'pnl'].mean()
        avgLoss = self.history(ticker).loc[self.history(ticker)['pnl'] < 0, 'pnl'].mean()
        # in case of nan values
        avgWin = avgWin if avgWin == avgWin else 0 
        avgLoss = avgLoss if avgLoss == avgLoss else 0
        return(
            - avgWin / avgLoss
        )

    def performance(self, benchmark=None):
        """
        Args:
            benchmark (str): Ticker symbol, e.g. BOVA11.SA, QQQ, SPY
        """ 
        colours = ['indigo', 'magenta', 'orangered', 'teal', 'crimson', 'mediumvioletred', 'rebeccapurple', 'seagreen']
        cnt = 0

        fig, axs = pl.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        for ticker in self._history['ticker'].unique():
            tickerResults = self._history[self._history['ticker']==ticker].copy()

            axs[0].plot(
                tickerResults['entry_date'],
                tickerResults['pnl'].cumsum().fillna(method='ffill'),
                label=f'{ticker}',
                linestyle='-',
                lw=2,
                color=colours[cnt % len(colours)]
            )
            cnt += 1
        # total pnl 
        dailyResults = pd.DataFrame(
            self._history.groupby(by='entry_date')['pnl'].sum().reset_index()
        )
        axs[0].plot(
                dailyResults['entry_date'],
                dailyResults['pnl'].cumsum().fillna(method='ffill'),
                label='Total PNL',
                linestyle='-',
                lw=2,
                color='black'
            )
        # add benchmark
        if benchmark:
            self._benchmark = pd.DataFrame(columns=['entry_date', 'pnl'])
            df = pdr.get_data_yahoo(
                benchmark,
                start=min(self._history['entry_date']),
                end=max(self._history['entry_date'])
            )
            self._benchmark['entry_date'] = self._history['entry_date'].unique()
            # merge with df
            self._benchmark = self._benchmark.merge(
                df['Close'],
                left_on='entry_date',
                right_index=True
            )
            self._benchmark['pnl'] = self._initial_cap * (self._benchmark['Close'] / self._benchmark.loc[0, 'Close'] - 1)

            axs[0].plot(
                self._benchmark['entry_date'],
                self._benchmark['pnl'],
                label=benchmark,
                linestyle='--',
                lw=1,
                color='gray'
            )
    
        axs[0].set_ylabel('Cumulative PNL (R$)')
        axs[0].legend()

        # pnl barplot
        axs[1].bar(
            dailyResults['entry_date'], 
            dailyResults['pnl'],
            label='PNL'
        )

        axs[1].legend()
        axs[1].set_xlabel('Entry date')
        axs[1].set_ylabel('PNL/Trade (R$)')
        
        pl.show()

    def plottrades(self, ticker, highAndLows=False):
        """Plot price history and trades on period for a stock ticker.

        Args:
            ticker (str): Stock symbol.
        """

        if highAndLows and not self._highNLows:
            # calculate high and lows
            for symbol in self.portfolio():
                topMask = (self.portfolio(symbol)['High'] > self.portfolio(symbol)['High'].shift(1)) & \
                    (self.portfolio(symbol)['High'] > self.portfolio(symbol)['High'].shift(-1))
                bottomMask = (self.portfolio(symbol)['Low'] < self.portfolio(symbol)['Low'].shift(1)) & \
                    (self.portfolio(symbol)['Low'] < self.portfolio(symbol)['Low'].shift(-1))
                self.portfolio(symbol).loc[topMask, 'Top'] = self.portfolio(symbol).loc[topMask, 'High']
                self.portfolio(symbol).loc[bottomMask, 'Bottom'] = self.portfolio(symbol).loc[bottomMask, 'Low']
            self._highNLows = True

        INCREASING_COLOR = '#17BECF'
        DECREASING_COLOR = '#7F7F7F'

        smaCols = self.portfolio(ticker).columns[self.portfolio(ticker).columns.str.contains('sma')]
        traces = []
        for col in smaCols:
            traces.append(
                go.Scatter(x=self.portfolio(ticker).index, y=self.portfolio(ticker)[col],
                    name=f'SMA {col[3:]}', line=dict(color='#E377C2'), yaxis='y2'),
            )

        if highAndLows:
            traces.append(
                go.Scatter(
                    x=self.portfolio(ticker).index, 
                    y=self.portfolio(ticker)['Top'],
                    name='Tops',
                    yaxis='y2',
                    mode='markers',
                    marker=dict(symbol='y-up', line=dict(width=2, color='orange'))  
                ),
            )
            traces.append(
                go.Scatter(
                    x=self.portfolio(ticker).index, 
                    y=self.portfolio(ticker)['Bottom'],
                    name='Bottoms',
                    yaxis='y2',
                    mode='markers',
                    marker=dict(symbol='y-down', line=dict(width=2, color='orange'))  
                ),
            )


        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=self.portfolio(ticker).index,
                    open=self.portfolio(ticker)['Open'], high=self.portfolio(ticker)['High'],
                    low=self.portfolio(ticker)['Low'], close=self.portfolio(ticker)['Close'],
                    increasing_line_color= INCREASING_COLOR, decreasing_line_color= DECREASING_COLOR,
                    name=ticker,
                    yaxis='y2'
                ),
                go.Scatter(
                    x=self.history(ticker)['entry_date'], 
                    y=self.history(ticker)['entry'],
                    name='Entry',
                    yaxis='y2',
                    mode='markers',
                    marker=dict(color='orange', symbol='triangle-up')    
                ),
                go.Scatter(
                    x=self.history(ticker)['entry_date'], 
                    y=self.history(ticker)['stop'],
                    name='Stop loss',
                    yaxis='y2',
                    mode='markers',
                    marker=dict(color='red', symbol='triangle-down')
                ),
                go.Scatter(
                    x=self.history(ticker)['entry_date'], 
                    y=self.history(ticker)['target'],
                    name='Target',
                    yaxis='y2',
                    mode='markers',
                    marker=dict(color='green', symbol='triangle-down')
                ),
                go.Bar(x=self.portfolio(ticker).index, y=self.portfolio(ticker)['Volume'], name='Volume', yaxis='y')
            ] + traces
        )

        fig.update_layout(
            title=ticker,
            yaxis2_title='R$',
            xaxis={
                'rangeslider_visible':False,
                "rangebreaks": [{
                    "bounds": ["sat", "mon"],
                    "values":["2015-12-24", "2015-12-25", "2016-01-01"]
                }]
            },
            yaxis1=dict(domain=[0, 0.2], showticklabels=False),
            yaxis2=dict(domain=[0.2, 0.8], type='log'),
            legend=dict(orientation='h', y=0.9, x=-0.7, yanchor='bottom'),
            margin=dict(t=40, b=40, r=40, l=40),
            plot_bgcolor='rgb(250, 250, 250)'
        )

        fig.show()