from re import template
import pandas as pd
from datetime import time, datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import notebook
import re
from utils import *

class Backtest:

    # updaterisk = False
    _timeframe = ''
    _trader = None
    _period = ()

    _highNLows = False
    _benchmark = pd.DataFrame(columns=[
        'entry_date',
        'pnl'
    ])
    _summary = pd.Series(index=[
        'ticker',
        'period',
        'strategy',
        'max_cap_needed',
        'total_trades',
        'open_trades',
        'gross_pnl',
        'total_fees',
        'liquid_pnl',
        'roi',
        'max_liq_pnl',
        'win_avg_time_in',
        'loss_avg_time_in',
        'efficiency',
        'sharpe',
        'pnl_ratio',
        'max_drawdown',
        'liquid_cap',
        'total_cap',
    ], dtype='float64', name='summary')

    def __init__(self, trader) -> None:
        self._trader = trader
    
    @property
    def timeframe(self):
        return(self._timeframe)
    
    @timeframe.setter
    def timeframe(self, timeFrame):
        assert re.match(r"^(daily|weekly|[0-9]+min)$", timeFrame.lower()), 'Timeframe not recognised...'
        self._timeframe = timeFrame

    @property
    def period(self):
        return(self._period)
    
    @period.setter
    def period(self, interval):
        """period stores two datetime variables in a tuple: (start, end)
        in the datetime format: (Y, m, d, H, M)

        Args:
            interval (tuple of strings): "DD-MM-YYYY"
        """
        startDD, startMM, startYYYY = (int(s) for s in interval[0].split('-'))
        endDD, endMM, endYYYY = (int(s) for s in interval[1].split('-'))
        if re.match(r"^([0-9]+min)$", self.timeframe.lower()):
            self._period = (datetime(startYYYY, startMM, startDD, 9, 0), datetime(endYYYY, endMM, endDD, 18, 0))
        elif re.match(r"^(daily)$", self.timeframe.lower()):
            self._period = (datetime(startYYYY, startMM, startDD, 10, 0), datetime(endYYYY, endMM, endDD, 10, 0))

    def run(self, start, end=datetime.today(), download=False):
        """Simulates trades over the chosen period

        Args:
            period (tuple | list): ('DD-MM-YYYY', 'DD-MM-YYYY')
        """
        self.period = (start, end)
        deltaPeriod = self.period[1] - self.period[0]
        self._trader.broker.load(start=self.period[0], end=self.period[1], timeseries=self.timeframe, download=download)
        deltat = timedelta(minutes=int(self.timeframe[:-3])) if re.match(r"^([0-9]+min)$", self.timeframe.lower()) else timedelta(days=1)

        self._trader.broker.calendarclock = self.period[0]
        currentDate = self._trader.broker.calendarclock.date()
        pbar1 = notebook.trange(deltaPeriod.days + 1, unit='days', desc='Running')
        while self._trader.broker.calendarclock <= self.period[1]:
            for strategy in self._trader.strategies:
                candleIdx = self._trader.broker.candleidx(strategy.ticker, self._trader.broker.calendarclock)
                if candleIdx > 0:
                    strategy.evaluate(candleIdx)
                    # print(candleIdx)
            
            # TODO: Include progress bar for time based on number of days
            self._trader.broker.calendarclock += deltat
            if self._trader.broker.calendarclock.time() > time(self._trader.broker.closedTimes[0], 0):
                self._trader.broker.calendarclock += timedelta(days=1)
                year, month, day = str(self._trader.broker.calendarclock.date()).split('-')
                self._trader.broker.calendarclock = datetime(int(year), int(month), int(day), self._trader.broker.closedTimes[1], 0)
            
            # update day
            if self._trader.broker.calendarclock.date() > currentDate:
                deltaDays = self._trader.broker.calendarclock.date() - currentDate
                currentDate = self._trader.broker.calendarclock.date() 
                pbar1.update(deltaDays.days)
            # print(self._trader.broker.calendarclock)

        pbar1.close()

    def pnl(self, ticker=None):
        """
        Calculates the profit or loss of the trade history
        """
        entryFinVol = self._trader.history(ticker)['entry'] * self._trader.history(ticker)['entry_vol']
        exitFinVol = self._trader.history(ticker)['exit'] * self._trader.history(ticker)['exit_vol']
        self._trader.history(ticker)['pnl'] = - (entryFinVol + exitFinVol)
        self._trader.history(ticker)['liquid_pnl'] = self._trader.history(ticker)['pnl'] - self._trader.history(ticker)['fees']
        
    def maxcapneeded(self, ticker=None):
        """Max cap needed to buy in a trade"""
        maxOnEntry = max(self._trader.history(ticker)['entry'] * self._trader.history(ticker)['entry_vol'])
        maxOnExit = max(self._trader.history(ticker)['exit'] * self._trader.history(ticker)['exit_vol'])
        return(max(maxOnEntry, maxOnExit))

    def totalpnl(self, ticker=None, liquid=False):
        """
        Calculates the total profit or loss of the trade history
        """
        assert (liquid is True) or (liquid is False), 'liquid shall be either True or False'
        self.pnl(ticker)
        self._trader.history(ticker)['total_pnl'] = self._trader.history(ticker)['pnl'].cumsum()
        self._trader.history(ticker)['total_liquid_pnl'] = self._trader.history(ticker)['liquid_pnl'].cumsum()
        totalPnlCol = 'liquid_pnl' if liquid else 'pnl'
        return(
            self._trader.history(ticker=ticker)[totalPnlCol].sum()
        )

    def totalfees(self, ticker=None):
        return(self._trader.history(ticker)['fees'].sum())

    def efficiency(self, ticker=None):
        """
        Calculates the ratio between number of winning trades to the total number of trades.
        """
        nOfTrades = self._trader.history(ticker)[self._trader.history(ticker)['status'] == 'closed'].shape[0]
        winningRate = round(100 * self._trader.history(ticker)[self._trader.history(ticker)['pnl'] > 0].shape[0] / nOfTrades, 2)
        losingRate = round(100 * self._trader.history(ticker)[self._trader.history(ticker)['pnl'] < 0].shape[0] / nOfTrades, 2)
        beRate = round(100 * self._trader.history(ticker)[self._trader.history(ticker)['pnl'] == 0].shape[0] / nOfTrades, 2)

        return(pd.Series({'winning': winningRate, 'losing': losingRate, 'break_even': beRate}))
    
    def avgtimein(self, winningTrades=False, losingTrades=False, ticker=None):
        """Calcultes the time in each trade and calculates the average.
        """
        self.pnl(ticker) if 'pnl' not in self._trader.history(ticker).columns else None
        self._trader.history(ticker)['entry_datetime'] = pd.to_datetime(self._trader.history(ticker)['entry_datetime'])
        self._trader.history(ticker)['exit_datetime'] = pd.to_datetime(self._trader.history(ticker)['exit_datetime'])
        self._trader.history(ticker)['time_in'] = self._trader.history(ticker)['exit_datetime'] - self._trader.history(ticker)['entry_datetime']
        
        if (winningTrades is True) and (losingTrades is False):
            mask = self._trader.history(ticker)['pnl'] > 0
            return(self._trader.history(ticker).loc[mask, 'time_in'].mean())
        elif (winningTrades is False) and (losingTrades is True):
            mask = self._trader.history(ticker)['pnl'] < 0
            return(self._trader.history(ticker).loc[mask, 'time_in'].mean())
        else:
            return(self._trader.history(ticker)['time_in'].mean())

    def sharpe(self, ticker=None):
        """
        Calculates the ratio between average profit to the average loss.
        """
        avgWin = self._trader.history(ticker).loc[self._trader.history(ticker)['pnl'] > 0, 'pnl'].mean()
        avgLoss = self._trader.history(ticker).loc[self._trader.history(ticker)['pnl'] < 0, 'pnl'].mean()
        # in case of nan values
        avgWin = avgWin if avgWin == avgWin else 0 
        avgLoss = avgLoss if avgLoss == avgLoss else 0
        return(- avgWin / avgLoss)

    def totalvol(self, before=None):
        """Calculates total volume of open trades

        Args:
            before (str): date in string format dd-mm-YYYY HH:MM        
        """
        before = strtodatetime(before) if before else self._trader.broker.calendarclock
        mask = self._trader.history()['entry_datetime'] <= before
        groupByTicker = self._trader.history()[mask].groupby('ticker')
        return(groupByTicker['entry_vol'].sum() + groupByTicker['exit_vol'].sum())

    def totalcap(self, atDateTime=None):
        """Calculates total capital on the date: atDateTime.
            totalcap = capital + sum(volume_ticker * price_atDate)

        Args:
            atDateTime (datetime | int): if -1 date is selected as last day with data available for the current ticker
        
        Return:
            cap (float): total capital calculated atDate.
        """
        atDateTime = strtodatetime(atDateTime) if atDateTime else self._trader.broker.calendarclock
        volDf = self.totalvol(atDateTime)
        cap = self._trader.capital
        for ticker in volDf.index:
            mask = self._trader.broker.portfolio[ticker].index <= atDateTime
            priceAtDate = self._trader.broker.portfolio[ticker][mask].iloc[-1]['Close']
            cap += volDf[ticker] * priceAtDate
        return(cap)

    def drawdown(self, ticker=None, inpercent=False):
        "Calculates the maximun draw down of the period considering the maximum capital needed to trade."
        maxDrawDown = 0
        for idx in self._trader.history(ticker).index:
            maxPl = self._trader.history(ticker).loc[self._trader.history(ticker).index <= idx, 'total_liquid_pnl'].max()
            minPl = self._trader.history(ticker).loc[self._trader.history(ticker).index > idx, 'total_liquid_pnl'].min()
            maxDrawDown = maxPl - minPl if maxPl - minPl > maxDrawDown else maxDrawDown
        maxDrawDown = round(100 * maxDrawDown / self.maxcapneeded(ticker), 2) if inpercent else maxDrawDown
        return(maxDrawDown)

    def pnlratio(self, ticker):
        '''Calculates the ratio of profit to loss.'''
        maskP = self._trader.history(ticker)['pnl'] > 0
        maskL = self._trader.history(ticker)['pnl'] < 0

        profit = self._trader.history(ticker).loc[maskP, 'pnl'].sum()
        loss = self._trader.history(ticker).loc[maskL, 'pnl'].sum()
        return(profit / abs(loss))

    def summary(self, ticker=None):
        """Returns results over the backtested period.
        """
        self._summary['ticker'] = ticker
        self._summary['period'] = '-'.join([d.strftime('%d-%m-%Y') for d in self.period])
        self._summary['strategy'] = self._trader.history(ticker)['strategy'].unique().tolist()
        self._summary['max_cap_needed'] = self.maxcapneeded(ticker)
        self._summary['win_avg_time_in'] = str(self.avgtimein(ticker=ticker, winningTrades=True))
        self._summary['loss_avg_time_in'] = str(self.avgtimein(ticker=ticker, losingTrades=True))
        self._summary['total_trades'] = self._trader.history(ticker).shape[0]
        self._summary['open_trades'] = self._trader.history(ticker)[self._trader.history(ticker)['status']=='close'].shape[0]
        self._summary['gross_pnl'] = self.totalpnl(ticker)
        self._summary['total_fees'] = self.totalfees(ticker)
        self._summary['liquid_pnl'] = self.totalpnl(ticker, liquid=True)
        self._summary['max_liq_pnl'] = self._trader.history(ticker)['total_liquid_pnl'].max()
        self._summary['roi'] = round(self._summary['liquid_pnl'] * 100 / self.maxcapneeded(ticker), 2)
        self._summary['efficiency'] = self.efficiency(ticker)['winning']
        self._summary['sharpe'] = self.sharpe(ticker)
        self._summary['pnl_ratio'] = self.pnlratio(ticker)
        self._summary['max_drawdown'] = self.drawdown(ticker, inpercent=True)
        self._summary['liquid_cap'] = self._trader.capital
        self._summary['total_cap'] = self.totalcap()


        idxs = self._summary.index[:-2] if ticker else self._summary.index
        return(self._summary[idxs])

    def pnldist(self, ticker=None, by='entry_datetime'):
        """Plot the pnl distrubution by hour for intraday strategies"""
        assert by in ['entry_datetime', 'exit_datetime']
        if self.timeframe == 'intraday':
            fig = go.Figure()

            fig.add_trace(go.Box(
                y=self._trader.history(ticker)['pnl'],
                x=self._trader.history(ticker)[by].dt.hour,
                name='Pnl',
                boxmean=True,
                boxpoints='all',
            ))
            fig.update_xaxes(categoryorder='category ascending')
            fig.update_layout(
                title_text='PNL distribution',
                template='plotly_dark', 
                height=600, 
                yaxis=dict(title='PNL per trade (R$)'),
                xaxis=dict(title='Trading hour')
            )
            fig.show()
        else:
            print('Available for intraday backtest only')

    # for strategy, performance should be changed
    def performance(self, benchmark=None, includeFees=False, intraday=False, printmode=False):
        """
        Args:
            benchmark (str): Ticker symbol, e.g. BOVA11.SA, QQQ, SPY
        """ 
        # colours = ['indigo', 'magenta', 'orangered', 'teal', 'crimson', 'mediumvioletred', 'rebeccapurple', 'seagreen']
        # cnt = 0
        theme = 'plotly_white' if printmode else 'plotly_dark'

        pnlCol = 'liquid_pnl' if includeFees is True else 'pnl'
        if intraday is True:
            df = self._trader.history()[['entry_datetime', pnlCol]].copy()
        else:
            df = self._trader.history()[['entry_datetime', pnlCol]].groupby(by=self._trader.history()['entry_datetime'].dt.date).sum().reset_index()
        
        df['pnl_curve'] = df[pnlCol].cumsum()

        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])

        fig.append_trace(go.Scatter(
            name='PL',
            x=df['entry_datetime'],
            y=df["pnl_curve"],
            marker_color='seagreen'), 
            row=1, col=1
        )

        profit = self._trader.history()[pnlCol] > 0
        loss = self._trader.history()[pnlCol] <= 0

        fig.append_trace(go.Bar(
            name='Profit',
            x=self._trader.history().loc[profit, 'entry_datetime'].index,
            y=self._trader.history().loc[profit, pnlCol],
            hovertext=self._trader.history().loc[profit, 'entry_datetime'],
            marker_color='seagreen'
        ), row=2, col=1)
        fig.append_trace(go.Bar(
            name='Loss',
            x=self._trader.history().loc[loss, 'entry_datetime'].index,
            y=self._trader.history().loc[loss, pnlCol],
            hovertext=self._trader.history().loc[loss, 'entry_datetime'],
            marker_color='crimson'
        ), row=2, col=1)

        fig.update_yaxes(rangemode="tozero", row=1, col=1)
        fig.update_yaxes(title_text="PNL (R$)", row=1, col=1)
        fig.update_yaxes(title_text="PNL per trade (R$)", row=2, col=1)
        fig.update_xaxes(rangebreaks= [{"bounds": ['sat', 'mon'], "pattern":'day of week', "values":["2021-09-07"]}], row=1, col=1)

        fig.update_layout(height=900, template=theme, title_text="Performance")

        # fig.show()
        
        return(fig)
        

        
        # df['pnl_curve'] = df[pnlCol].cumsum()
        # fig1 = go.Figure()
        # fig1.add_trace(go.Scatter(
        #     x=df['entry_datetime'],
        #     y=df["pnl_curve"],
        #     marker_color='seagreen')
        # )
        # fig1.update_yaxes(rangemode="tozero")
        # fig1.update_layout(
        #     title_text='PNL over time',
        #     yaxis=dict(title='PNL (R$)'),
        #     height=500,
        #     template='plotly_dark', 
        #     xaxis={
        #         "rangebreaks": [{"bounds": ['sat', 'mon'], "pattern":'day of week', "values":["2021-09-07"]}],
        #     }
        # )
        # fig1.show()

        # profit = self._trader.history()[pnlCol] > 0
        # loss = self._trader.history()[pnlCol] <= 0

        # fig2 = go.Figure()
        # fig2.add_trace(go.Bar(
        #     name='Profit',
        #     x=self._trader.history().loc[profit, 'entry_datetime'].index,
        #     y=self._trader.history().loc[profit, pnlCol],
        #     hovertext=self._trader.history().loc[profit, 'entry_datetime'],
        #     marker_color='seagreen'
        # ))
        # fig2.add_trace(go.Bar(
        #     name='Loss',
        #     x=self._trader.history().loc[loss, 'entry_datetime'].index,
        #     y=self._trader.history().loc[loss, pnlCol],
        #     hovertext=self._trader.history().loc[loss, 'entry_datetime'],
        #     marker_color='crimson'
        # ))
        # fig2.update_layout(height=400, template='plotly_dark', yaxis=dict(title='PNL per trade (R$)'))

        # fig2.show()

    # Should be changed
    def plottrades(self, ticker, indicators=None, printmode=False):
        """Plot price history and trades on period for a stock ticker.

        Args:
            ticker (str): Stock symbol.
            indicators (list): list of dictionaries {'name': 'rsi', 'axis': 'same | new', 'color':'crimson', 'type':'line | bar'}
        """
        theme = 'plotly_white' if printmode else 'plotly_dark'
        candleColors = {'increasing': 'seagreen', 'decreasing': 'red'} if printmode else {'increasing': 'white', 'decreasing': 'black'}

        
        period = (self._trader.broker.portfolio[ticker].index >= self.period[0]) & (self._trader.broker.portfolio[ticker].index <= self.period[1])
        historyToPlot = self._trader.history(ticker).copy()
        dataToPlot = self._trader.broker.portfolio[ticker][period].copy()
        
        trades = maketradesplot(historyToPlot, axis='y2')
        timeseries = makecandlesticks(dataToPlot, ticker=ticker, axis='y2', colors=candleColors)
        indicatorsPlots = []
        for indicator in indicators if indicators is not None else []:
            indicator['axis'] = 'y2' if indicator['axis'] == 'same' else 'y'
            indicatorsPlots.append(
                makeindicatorplot(dataToPlot, indicator=indicator)
            )

        fig = go.Figure(data=[timeseries] + trades + indicatorsPlots)

        bounds = [{
            "bounds": ["sat", "mon"],
            "values":["2021-09-07", "2015-12-25", "2016-01-01"]
        }]

        bounds = bounds + [{"bounds": self._trader.broker.closedTimes, "pattern": 'hour'}] if len(dataToPlot.index.minute.unique()) > 1 else bounds

        fig.update_layout(
            template=theme,
            height=700,
            spikedistance=1000,
            hoverdistance=10,
            title=ticker,
            xaxis={
                'rangeslider_visible':False,
                "rangebreaks": bounds
            },
            yaxis1=dict(side='right', domain=[0, 0.15], showticklabels=True, showspikes=False, showline=False, spikecolor="grey", spikesnap='cursor', spikemode='across', spikedash='solid', spikethickness=1),
            yaxis2=dict(side='right', title='R$', domain=[0.15, 0.8], type='log', showspikes=False, showline=False, spikecolor="grey", spikesnap='cursor', spikemode='across', spikedash='solid', spikethickness=1),
            legend=dict(orientation='h', y=0.9, x=0.3, yanchor='bottom'),
            margin=dict(t=40, b=40, r=40, l=40),
        )

        return(fig)