import pandas as pd
import numpy as np
from datetime import time
from utils import *
# Improvement: add method to calculate stop and target
# Improvement: queue orders to be triggered or not. 

class Strategy:

    _trader = {}
    
    def bind(self, trader):
        "Give access to the traders data and methods"
        self._trader = trader

    @property
    def settings(self):
        """Adjusts the model using a dictionary to set values to the attributes of the class.
        """
        for k in self.__dir__():
            if (k!='settings') and (not k.startswith('_')) and (not callable(k)):
                v = getattr(self, k)
                if type(v) in (str, int, float, time):
                    print(f'{k}: {v}')
        return(None)
        
    @settings.setter
    def settings(self, _):
        """Adjusts the model using a dictionary to set values to the attributes of the class.
        """
        print('To change settings, call each parameter as self.<parameter> = <value>')
  

class ABC:
    """ABC on close.
    Rules:
        + Entry above signal (S.High) after C or above B.High
        + Stop bellow C.Low
        + Target fixed according to RR 
    """
    _pattern = pd.DataFrame()
    debug = False
    RRLowerThan1 = False
    stop = {'position': 'C', 'factor': 1}
    target = {'projection': 'AB', 'factor': 1} # 'AB' or None 
    selectCriteria = None # 4SMA, SMA21...
    name = 'ABC'

    _ticker = 'MGLU3'
    _trader = {}  # trader object

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.reset(trader._portfolio[self._ticker].iloc[0])
        if self.selectCriteria is None:
            pass
        elif self.selectCriteria.lower() == '4sma':
            self._trader._portfolio[ticker]['sma4'] = self._trader._portfolio[ticker]['Close'].rolling(window=4).mean()
            self._trader._portfolio[ticker]['sma17'] = self._trader._portfolio[ticker]['Close'].rolling(window=17).mean()
            self._trader._portfolio[ticker]['sma34'] = self._trader._portfolio[ticker]['Close'].rolling(window=34).mean()
            self._trader._portfolio[ticker]['sma80'] = self._trader._portfolio[ticker]['Close'].rolling(window=80).mean()
        elif self.selectCriteria.lower()[:3] == 'sma':
            smaPeriod = int(self.selectCriteria.lower()[3:])
            self._trader._portfolio[ticker][f'sma{smaPeriod}'] = self._trader._portfolio[ticker]['Close'].rolling(window=smaPeriod).mean()
        else:
            print('Selection criteria not available. Try None or 4sma.')
    
    @classmethod
    def settings(cls, attributesDict=None):
        """Adjusts the model using a dictionary to set values to the attributes of the class.
        """
        if attributesDict is None:
            settings = {}
            for k, v in cls.__dict__.items():
                if not (callable(cls.__dict__[k])) and (k!='settings') and not (k.startswith('_')):
                    # print(f'{k}: {v}')
                    settings[k] = v
            return(settings)
        else:
            for k, v in attributesDict.items():
                setattr(cls, k, v)

    def reset(self, initialCandle):
        self._pattern = pd.DataFrame(initialCandle).transpose()
        self._pattern['pivot'] = 'A'

    def add(self, pivot, candle):
        candle = candle.copy()
        candle['pivot'] = pivot
        self._pattern = self._pattern.append(candle)

    def updatepattern(self, candle):
        
        prevCandle = self._pattern.iloc[-1]
        if prevCandle['pivot'] == 'A':
            if (candle['Close'] <= prevCandle['Close']) or (candle['Low'] < prevCandle['Low']):
                self.reset(candle)
                msg = 'a down -> a'
            else:
                self.add('B', candle)
                msg = 'a up -> b'
        elif prevCandle['pivot'] == 'B':
            if (candle['Low'] < min(self._pattern['Low'])):
                self.reset(candle)
                msg = 'b min -> a'
            elif (candle['Close'] >= prevCandle['Close']) or ((candle['Close'] < prevCandle['Close']) and (candle['High'] > prevCandle['High'])):
                self.add('B', candle)
                msg = 'b up -> b'
            # elif (candle['Low'] >= min(self._pattern['Low'])):
            else:
                self.add('C', candle)
                msg = 'b down -> c'
            # else:
            #     self.reset(candle)
            #     msg = 'b min -> a'
        elif prevCandle['pivot'] == 'C':
            if (candle['Low'] < min(self._pattern['Low'])):
                self.reset(prevCandle)
                msg = 'c min -> a'
            elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] <= max(self._pattern['High'])):
                self.add('S', candle)
                msg = 'c up -> s'
            elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] > max(self._pattern['High'])):
                # trigger on B high
                self.reset(prevCandle)
                self.add('B', candle)
                msg = 'c up -> trigger on b'
            elif (candle['Close'] < prevCandle['Close']):
                self.add('C', candle)
                msg = 'c down -> c'
            else:
                print('Condition not taken into account')
        elif prevCandle['pivot'] == 'S':
            if candle['Low'] < min(self._pattern['Low']):
                self.reset(candle)
                msg = 's down -> a'
            # elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] > prevCandle['High']):
            # it will erase the B entry in case it doesn't trigger on this candle
            elif (candle['High'] > prevCandle['High']):
                # S is B
                self.reset(self._pattern.iloc[-2])
                self.add('B', prevCandle)
                self.add('B', candle)
                msg = 's up -> a,b,b'
            elif (candle['Close'] >= prevCandle['Close']):
                self.add('S', candle)
                msg = 's up -> s'
            else:
                self.add('C', candle)
                msg = 's down -> c'
            # else:
            #     self.reset(candle)
            #     msg = 's down -> a'
        
        if self.debug:
            print(f'{candle.name}: {self._ticker} {msg}')
        
    def criteria(self, candle):
        if self.selectCriteria is None:
            return(True)
        elif self.selectCriteria.lower() == '4sma':
            if (candle['sma4'] > candle['sma17'] > candle['sma34'] > candle['sma80']):
                return(True)
            else: return(False)
        elif self.selectCriteria.lower()[:3] == 'sma':
            if candle['Close'] > candle[self.selectCriteria.lower()]:
                return(True)
            
    def checktrigger(self, candle):
        """Checks if an entry is triggered on candle based on pattern.

        Args:
            candle (pd.Series): Time series pandas series
        return (float | None, float | None): entry and stop
        """
        entry, stop = None, None
        if self._pattern.iloc[-1]['pivot'] == 'S':
            minC = min(self._pattern.loc[self._pattern['pivot']=='C', 'Low'])
            tradingRange = {
                'min': min(self._pattern.iloc[-1]['Low'], minC),
                'max1': self._pattern.iloc[-1]['High'], # S high or max of S and C?
                'max2': max(self._pattern['High'])
            }
            if self.stop['position'] == 'C':
                stop = round(tradingRange['min'] - 0.01, 2)
            elif self.stop['position'] == 'A':
                stop = round(self._pattern['Low'].min() - 0.01, 2)
            else:
                print('Stop position not recognized, try: A or C')
                return(None)
            # if triggers on S
            if (
                (tradingRange['min'] <= candle['Open'] <= tradingRange['max1'])\
                and (tradingRange['max1'] < candle['High'])
            ) \
            or (
                (tradingRange['max1'] < candle['Open'] <= tradingRange['max2']) \
                and (candle['High'] <= tradingRange['max2']) \
                and (candle['Low'] <= tradingRange['max1'] + 0.01)
            ):
                entry = round(tradingRange['max1'] + 0.01, 2)
                # factor correction: 1 makes no change
                stop = entry - self.stop['factor'] * (entry - stop)
                
            # if triggers on C
            elif (
                (tradingRange['max1'] + 0.01 < candle['Open'] <= tradingRange['max2']) \
                and (candle['High'] > tradingRange['max2'])
            ) or (
                (candle['Open'] > tradingRange['max2']) \
                and (candle['Low'] <= tradingRange['max2'] + 0.01) 
            ):
                entry = round(tradingRange['max2'] + 0.01, 2)
                # factor correction: 1 makes no change
                stop = entry - self.stop['factor'] * (entry - stop)
                
        if self._pattern.iloc[-1]['pivot'] == 'C':
            tradingRange = {
                'min': min(self._pattern.loc[self._pattern['pivot']=='C', 'Low']),
                'max': max(self._pattern['High'])
            }
            if self.stop['position'] == 'C':
                stop = round(tradingRange['min'] - 0.01, 2)
            elif self.stop['position'] == 'A':
                stop = round(self._pattern['Low'].min() - 0.01, 2)
            else:
                print('Stop position not recognized, try: A or C')
                return(None)
            
            if ((tradingRange['min'] <= candle['Open']) \
                and (tradingRange['max'] <= candle['High'] + 0.01) and (candle['Low'] <= tradingRange['max'] + 0.01)):
                entry = round(tradingRange['max'] + 0.01, 2)
                # factor correction: 1 makes no change
                stop = entry - self.stop['factor'] * (entry - stop)
        
        return(entry, stop)

    def entrycallback(self, idx):
        """
        """
        candle = self._trader._portfolio[self._ticker].iloc[idx]
        entry, stop = self.checktrigger(candle)

        if entry and stop:
            if self.criteria(self._trader._portfolio[self._ticker].iloc[idx - 1]):
                # filter RR lower than 1
                if self.target['projection'] == 'AB':
                    target = (self._pattern['High'].max() - self._pattern['Low'].min()) * self.target['factor'] + entry
                else:
                    target = None
                if (target is None) or (self.RRLowerThan1) or ((target - entry) > (entry - stop)):
                    print(f'{candle.name}: {self._ticker} {entry=}, {stop=}')
                    tradeID = self._trader.buy(ticker=self._ticker, entry=entry,
                                        stop=stop, entry_date=candle.name, strategy_name=self.name)
                    self._trader.settarget(tradeID, target=target)

        self.updatepattern(candle)
    
    def exitcallback(self, dayIdx):

        candle = self._trader._portfolio[self._ticker].iloc[dayIdx]
        
        boughtFilter = (self._trader._history['status'] == 'bought') & (
            self._trader._history['ticker'] == self._ticker)
        for idx, position in self._trader._history[boughtFilter].iterrows():
            price = None
            atDate = None
            # if candle['High'] >= position['target']:
            # Consider gaps
            if candle['High'] >= position['target']:
                if candle['Open'] >= position['target']:
                    price = round(candle['Open'], 2)
                else:
                    price = round(position['target'], 2)

                atDate = candle.name
                if position['target'] > position['entry']:
                    print('Profit!')
                else:
                    print('Loss...')

            elif candle['Low'] <= position['stop']:
                if candle['Open'] <= position['stop']:
                    price = round(candle['Open'], 2)
                else:
                    price = round(position['stop'], 2)
                atDate = candle.name
                print('Loss...')

            if price and atDate:
                self._trader.sell(price, atDate, idx)
                print(f'{atDate}: {self._ticker} {price=}')

    # def abcshape(self):
    #     """
    #     """

    #     abHeight = self._pattern['High'].max() - self._pattern['Low'].min()
    #     abGain = abHeight / self._pattern['Low'].min()
    #     retraceLv = (self._pattern['High'].max() - self._pattern.loc[self._pattern['pivot']=='C', 'Low'].min()) / abHeight
        
    #     bcHeight = round(

    #     )
    #     return(abHeight, bcHeight)

    def searchcallback(self, idx):
        """
        """
        candle = self._trader._portfolio[self._ticker].iloc[idx]
        self.updatepattern(candle)

        minC = self._pattern.loc[self._pattern['pivot']=='C', 'Low'].min()
        
        if self.criteria(candle):
            if self._pattern.iloc[-1]['pivot'] == 'S':
                entry = round(self._pattern.iloc[-1]['High'] - 0.01, 2)
                stop = round(min(minC, self._pattern.iloc[-1]['Low']) - 0.01, 2)
                triggerOn = 'S'
                # print(self._ticker, entry, stop)
            elif self._pattern.iloc[-1]['pivot'] == 'C':
                entry = round(self._pattern['High'].max() - 0.01, 2)
                stop = round(minC - 0.01, 2)
                triggerOn = 'C'
                # print(self._ticker, entry, stop)
            else:
                entry, stop, triggerOn = None, None, None
        else:
            entry, stop, triggerOn = None, None, None
        stop = None if stop is None else entry - self.stop['factor'] * (entry - stop)
        # print(self._ticker, entry, stop)
        return(entry, stop, triggerOn)


class RSIN(Strategy):
    """RSIN (updated): Relative strength index with N periods
    Currently developed for daily 

    Rules:
        + Entry on close of the bar when RSI-N < rsi limit
        + Stop: on time, N bars with no profit, or predefined
        + Target max of the high of previous 2 bars
    """

    debug = False
    _trader = {}  # trader object
    # _settings = {}
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.rsiPeriod = 2
        self.nTargetCandles = 2 # min 1
        self.name = 'RSI-N'
        self.rsiLimit = 20
        # self.smaPeriod = 5
        self.quantity = 400
        # self.stopGain = 0.14
        self.stopLoss = 3.0
        self.stopInTime = 7
        # print('Do not change <nTargetCandles>. The code has to be improved.\n')

    def request(self):
        """Requests any calculations to the broker"""
        # self._trader.broker.sma(self.smaPeriod)
        self._trader.broker.rsi(self.rsiPeriod)

    def evaluate(self, idx):
        """
        """
        candle = self._trader.broker.portfolio[self.ticker].iloc[idx]
        prevCandles = self._trader.broker.portfolio[self.ticker].iloc[idx - self.nTargetCandles:idx-1]
        # targetCandle = self._trader.broker.portfolio[self.ticker].iloc[idx - self.nTargetCandles:-1]
        # print(candle.name)
        # print(prevCandles)
        # print('')

        isOpen = self._trader.isBought(self.ticker)
        if isOpen.any():
            trade = self._trader.history(self.ticker)[isOpen]
            assert trade.shape[0] == 1, f'More than 1 position open for {self.ticker}'
            # print(trade)
            candleIdx = self._trader.broker.candleidx(self.ticker, trade['entry_datetime'].item())
            # Check for stop in time
            if (idx > candleIdx + self.stopInTime):
                self._trader.close(trade.index.item(), candle['Open'], candle.name, self.name)
            # check stop loss on same entry candle
            elif (idx == candleIdx):
                print('Error on strategy...')           
            else:
                # sell stop on stop loss and sell limit the high of previous 2 candles
                self._trader.sell(self.ticker, self.quantity, trade['stop'].item(), candle.name, orderType='stop', strategyName=self.name)
                isOpen = self._trader.isBought(self.ticker)
                # if it is still open
                if isOpen.any():
                    # sellPrice = max(targetCandle['High'], prevCandle['High'])
                    sellPrice = prevCandles['High'].max()
                    self._trader.sell(self.ticker, self.quantity, sellPrice, candle.name, orderType='limit', strategyName=self.name)
                
        if (candle[f'RSI{self.rsiPeriod}'] < self.rsiLimit) and (not self._trader.isBought(self.ticker).any()):
            buyPrice = candle['Close']
            stopLoss = buyPrice - self.stopLoss
            # stopGain = max(candle['High'], prevCandle['High'])
            stopGain = max(prevCandles['High'].iloc[1:].max(), candle['High'])

            tradeIdx = self._trader.buy(self.ticker, self.quantity, buyPrice, candle.name, orderType='market', strategyName=self.name)
        else:
            tradeIdx = None

        if (tradeIdx is not None) and tradeIdx >= 0: 
            self._trader.setstop(tradeIdx, stopLoss)
            self._trader.settarget(tradeIdx, stopGain)
         

class HighNLows:
    """High-and-Lows: Strategy developed for daytrade (Long or Short)
    
    Rules:
        + Entry long on low of the bar if its close is above the SMA or short on the high if its close is below the SMA 
        + Stop: on time, N bars with no profit, or predefined
        + Target predefined or when there is an oposite signal 

        + filter: above SMA-P, simple moving avarage of P periods
    """

    name = 'High-and-Lows'
    
    _trader = {}  # trader object
    # _settings = {}
    _start = None
    _end = None

    def __init__(self, ticker):
        self.ticker = ticker
        self.quantity = 400
        self.start = '10:12'
        self.end = '16:30'
        self.stopGain = 0.14
        self.stopLoss = 0.25
        self.stopInTime = 4
        self.smaPeriod = 5
    
    def request(self):
        """Requests any calculations to the broker"""
        self._trader.broker.sma(self.smaPeriod)

    def evaluate(self, idx):
        """
        """
        candle = self._trader.broker.portfolio[self.ticker].iloc[idx]
        prevCandle = self._trader.broker.portfolio[self.ticker].iloc[idx - 1]
        # Check long or short signals
        if (prevCandle[f'Close'] > prevCandle[f'sma{self.smaPeriod}']):
            # long signal
            buyPrice = prevCandle['Low']
            sellPrice = None
        elif (prevCandle[f'Close'] < prevCandle[f'sma{self.smaPeriod}']):
            # short signal
            sellPrice = prevCandle['High']
            buyPrice = None
        else:
            sellPrice = None
            buyPrice = None
        # print(candle.name)
        # print(prevCandle.name)
        # print('')

        isOpen = self._trader.isBought(self.ticker) | self._trader.isSold(self.ticker)
        if isOpen.any():
            trade = self._trader.history(self.ticker)[isOpen]
            tradeSide = trade['side'].item().lower()
            assert trade.shape[0] == 1
            # print(trade)
            candleIdx = self._trader.broker.candleidx(self.ticker, trade['entry_datetime'].item())
            # Check for stop in time and close at the end of the day
            if (idx > candleIdx + self.stopInTime) or (self._trader.broker.calendarclock.time() >= self.end):
                self._trader.close(trade.index.item(), candle['Open'], candle.name, self.name)
            # check stop gain and loss on same entry candle
            elif (idx == candleIdx):
                print(idx, candleIdx)
                delta = candle['Close'] - trade['entry'] if tradeSide.lower() == 'long' else -(candle['Close'] - trade['entry'])
                delta = delta.item()
                if (delta >= self.stopGain) or (- delta >= self.stopLoss):
                    self._trader.close(trade.index.item(), candle['Close'], candle.name, self.name)
                    return() # return to skip any other action on same bar            
            else:
                if (tradeSide == 'long'):
                    self._trader.sell(self.ticker, self.quantity, trade['stop'].item(), candle.name, orderType='stop', strategyName=self.name)
                    if (sellPrice is None) or (sellPrice > trade['target'].item()):
                        self._trader.sell(self.ticker, self.quantity, trade['target'].item(), candle.name, orderType='limit', strategyName=self.name)
                        return()
                elif (tradeSide == 'short'):
                    self._trader.buy(self.ticker, self.quantity, trade['stop'].item(), candle.name, orderType='stop', strategyName=self.name)
                    if (buyPrice is None) or (buyPrice < trade['target'].item()):
                        self._trader.buy(self.ticker, self.quantity, trade['target'].item(), candle.name, orderType='limit', strategyName=self.name)
                        return()

        if (self.start <= self._trader.broker.calendarclock.time() < self.end):
            if (buyPrice is not None) and (not self._trader.isBought(self.ticker).any()):
                stopLoss = buyPrice - self.stopLoss
                stopGain = buyPrice + self.stopGain
                qnt = 2 * self.quantity if (self._trader.isSold(self.ticker).any()) else self.quantity
                tradeIdx = self._trader.buy(self.ticker, qnt, buyPrice, candle.name, orderType='limit', strategyName=self.name)
                
            elif (sellPrice is not None) and (not self._trader.isSold(self.ticker).any()):
                stopGain = sellPrice - self.stopGain
                stopLoss = sellPrice + self.stopLoss 
                qnt = 2 * self.quantity if (self._trader.isBought(self.ticker).any()) else self.quantity
                tradeIdx = self._trader.sell(self.ticker, qnt, sellPrice, candle.name, orderType='limit', strategyName=self.name)
            else:
                tradeIdx = None

            if (tradeIdx is not None) and tradeIdx >= 0: 
                self._trader.setstop(tradeIdx, stopLoss)
                self._trader.settarget(tradeIdx, stopGain)

    def bind(self, trader):
        "Give access to the traders data and methods"
        self._trader = trader

    def _strtotime(self, timestr):
        h, m = (int(n) for n in timestr.split(':'))
        t = time(h, m)
        return(t)

    @property
    def start(self):
        return(self._start)

    @start.setter
    def start(self, hhmm):
        self._start = self._strtotime(hhmm)
    
    @property
    def end(self):
        return(self._end)

    @end.setter
    def end(self, hhmm):
        self._end = self._strtotime(hhmm)

    @property
    def settings(self):
        """Adjusts the model using a dictionary to set values to the attributes of the class.
        """
        for k in self.__dir__():
            if (k!='settings') and (not k.startswith('_')) and (not callable(k)):
                v = getattr(self, k)
                if type(v) in (str, int, float, time):
                    print(f'{k}: {v}')
        return(None)
        
    @settings.setter
    def settings(self, _):
        """Adjusts the model using a dictionary to set values to the attributes of the class.
        """
        print('To change settings, call each parameter as self.<parameter> = <value>')
       

class OTT(Strategy):
    """One two three

    Rules:
        + Entry on the break of a 123 pattern: tops or bottoms for long or shorts
        + A slow and a fast MA is used as filters
        + Stop: on the tip the centre candle
        + as a projection of the size of the pattern

    """

    debug = False
    
    def __init__(self, ticker):
        self.name = '123'
        self.ticker = ticker
        self.slowMA = 'sma80'
        self.fastMA = 'sma8'
        self.targetProjection = 1.6
        self.stopProjection = 1.0
        self.breakOutDelta = 0.01
        self.fixedRisk = 'NA' # in cash amount
        self.quantity = 400

    def request(self):
        """Requests any calculations to the broker"""
        getattr(self._trader.broker, self.slowMA[:3])(period=int(self.slowMA[3:]))
        getattr(self._trader.broker, self.fastMA[:3])(period=int(self.fastMA[3:]))

    def evaluate(self, idx):
        """
        """
        candle = self._trader.broker.portfolio[self.ticker].iloc[idx]
        candle3 = self._trader.broker.portfolio[self.ticker].iloc[idx - 1]
        candle2 = self._trader.broker.portfolio[self.ticker].iloc[idx - 2]
        candle1 = self._trader.broker.portfolio[self.ticker].iloc[idx - 3]

        # isTop = (candle2['High'].item() > candle1['High'].item()) and (candle2['High'].item() > candle3['High'].item())
        isBottom = (candle2['Low'].item() < candle1['Low'].item()) and (candle2['Low'].item() < candle3['Low'].item())
        

        isOpen = self._trader.isBought(self.ticker)
        if isOpen.any():
            trade = self._trader.history(self.ticker)[isOpen]
            assert trade.shape[0] == 1, f'More than 1 position open for {self.ticker}'
            # print(trade)
            candleIdx = self._trader.broker.candleidx(self.ticker, trade['entry_datetime'].item())
            
            if (idx == candleIdx):
                print('Error on strategy...')           
            else:
                # sell stop
                stopped = self._trader.sell(self.ticker, self.quantity, trade['stop'].item(), candle.name, orderType='stop', strategyName=self.name)
                isOpen = self._trader.isBought(self.ticker)
                # if it is still open
                if isOpen.any():
                    targetted = self._trader.sell(self.ticker, self.quantity, trade['target'].item(), candle.name, orderType='limit', strategyName=self.name)
                if not self._trader.isBought(self.ticker).any():
                    return

        if (candle3[self.fastMA] >= candle2[self.fastMA]) and \
            (candle3[self.slowMA] >= candle2[self.slowMA]) and \
            (candle3[self.fastMA] > candle3[self.slowMA]) and \
            isBottom and \
            (not self._trader.isBought(self.ticker).any()):
            
            risk = candle3['High'] - candle2['Low']
            buyPrice = candle3['High'] + self.breakOutDelta
            stopLoss = candle3['High'] - self.stopProjection * risk - self.breakOutDelta
            stopGain = self.targetProjection * risk + buyPrice
            self.quantity = int(self.fixedRisk / risk) if (self.fixedRisk != 'NA') else self.quantity
            tradeIdx = self._trader.buy(self.ticker, self.quantity, buyPrice, candle.name, orderType='stop', strategyName=self.name)
        else:
            tradeIdx = None

        if (tradeIdx is not None) and tradeIdx >= 0: 
            self._trader.setstop(tradeIdx, stopLoss)
            self._trader.settarget(tradeIdx, stopGain)
            

class EMACross(Strategy):
    """EMA

    Rules:
        + Entry on the break of a 123 pattern: tops or bottoms for long or shorts
        + A slow and a fast MA is used as filters
        + Stop: on the tip the centre candle
        + as a projection of the size of the pattern

    """

    def __init__(self, ticker):
        self.name = 'EMA-cross'
        self.ticker = ticker
        self.slowMA = 'ema21'
        self.fastMA = 'ema3'
        self.delta = 5
        self.target = 10
        self.stopLoss = 2
        self.quantity = 100
        self.start = '10:12'
        self.end = '16:30'

    def request(self):
        """Requests any calculations to the broker"""
        getattr(self._trader.broker, self.slowMA[:3])(period=int(self.slowMA[3:]), at='Close')
        getattr(self._trader.broker, self.fastMA[:3])(period=int(self.fastMA[3:]), at='Low')
        getattr(self._trader.broker, self.fastMA[:3])(period=int(self.fastMA[3:]), at='High')

    @property
    def start(self):
        return(self._start)

    @start.setter
    def start(self, hhmm):
        self._start = strtotime(hhmm)
    
    @property
    def end(self):
        return(self._end)

    @end.setter
    def end(self, hhmm):
        self._end = strtotime(hhmm)

    def evaluate(self, idx):
        """
        """
        candle = self._trader.broker.portfolio[self.ticker].iloc[idx]
        prevCandle = self._trader.broker.portfolio[self.ticker].iloc[idx - 1]
        intervalOfNCandles = self._trader.broker.portfolio[self.ticker].iloc[idx - self.delta : idx]
        crossHigh = (intervalOfNCandles[f'{self.fastMA}-High'] > intervalOfNCandles[self.slowMA]) &\
             (intervalOfNCandles[f'{self.fastMA}-High'].shift(1) < intervalOfNCandles[self.slowMA].shift(1))
                
        isOpen = self._trader.isBought(self.ticker)
        if isOpen.any():
            trade = self._trader.history(self.ticker)[isOpen]
            assert trade.shape[0] == 1, f'More than 1 position open for {self.ticker}'
            # print(trade)
            candleIdx = self._trader.broker.candleidx(self.ticker, trade['entry_datetime'].item())
            if (self._trader.broker.calendarclock.time() >= self.end):
                self._trader.close(trade.index.item(), candle['Open'], candle.name, self.name)
            elif (idx == candleIdx):
                print('Error on strategy...')           
            else:
                # sell stop
                self._trader.sell(self.ticker, self.quantity, trade['stop'].item(), candle.name, orderType='stop', strategyName=self.name)
                isOpen = self._trader.isBought(self.ticker)
                # if it is still open
                if isOpen.any():
                    self._trader.sell(self.ticker, self.quantity, trade['target'].item(), candle.name, orderType='limit', strategyName=self.name)
                if not self._trader.isBought(self.ticker).any():
                    return
                    
        if (self.start <= self._trader.broker.calendarclock.time() < self.end):
            if (prevCandle[f'{self.fastMA}-Low'] < prevCandle[self.slowMA]) and (candle[f'{self.fastMA}-Low'] > candle[self.slowMA]) \
                and (crossHigh.any()) and (not self._trader.isBought(self.ticker).any()):
                buyPrice = candle['Close']
                stopLoss = buyPrice - self.stopLoss
                target = buyPrice + self.target
                quantity = self.quantity
                tradeIdx = self._trader.buy(self.ticker, quantity, buyPrice, candle.name, orderType='market', strategyName=self.name)
            else:
                tradeIdx = None

            if (tradeIdx is not None) and tradeIdx >= 0: 
                self._trader.setstop(tradeIdx, stopLoss)
                self._trader.settarget(tradeIdx, target)
            


 # def bind(self, trader):
    #     "Give access to the traders data and methods"
    #     self._trader = trader

    # @property
    # def settings(self):
    #     """Adjusts the model using a dictionary to set values to the attributes of the class.
    #     """
    #     for k in self.__dir__():
    #         if (k!='settings') and (not k.startswith('_')) and (not callable(k)):
    #             v = getattr(self, k)
    #             if type(v) in (str, int, float, time):
    #                 print(f'{k}: {v}')
    #     return(None)
        
    # @settings.setter
    # def settings(self, _):
    #     """Adjusts the model using a dictionary to set values to the attributes of the class.
    #     """
    #     print('To change settings, call each parameter as self.<parameter> = <value>')

# class INSIDEBAR:
#     """Inside bar.
#     Rules:
#         + Entry above inside bar
#         + Stop bellow inside bar
#         + Target fixed according to RR 
#     """
#     debug = False
#     stop = {'position': 'C', 'factor': 1}
#     target = {'projection': 'AB', 'factor': 1} # 'AB' or None 
#     selectCriteria = None # 4SMA, SMA21...
#     name = 'inside bar'

#     _ticker = 'MGLU3'
#     _trader = {}  # trader object

#     def __init__(self, ticker, trader):
#         self._ticker = ticker
#         self._trader = trader
#         # self.reset(trader._portfolio[self._ticker].iloc[0])
#         if self.selectCriteria is None:
#             pass
#         elif self.selectCriteria.lower() == '4sma':
#             self._trader._portfolio[ticker]['sma4'] = self._trader._portfolio[ticker]['Close'].rolling(window=4).mean()
#             self._trader._portfolio[ticker]['sma17'] = self._trader._portfolio[ticker]['Close'].rolling(window=17).mean()
#             self._trader._portfolio[ticker]['sma34'] = self._trader._portfolio[ticker]['Close'].rolling(window=34).mean()
#             self._trader._portfolio[ticker]['sma80'] = self._trader._portfolio[ticker]['Close'].rolling(window=80).mean()
#         elif self.selectCriteria.lower()[:3] == 'sma':
#             smaPeriod = int(self.selectCriteria.lower()[3:])
#             self._trader._portfolio[ticker][f'sma{smaPeriod}'] = self._trader._portfolio[ticker]['Close'].rolling(window=smaPeriod).mean()
#         else:
#             print('Selection criteria not available. Try None or 4sma.')
    
#     # @classmethod
#     # def settings(cls, attributesDict=None):
#     #     """Adjusts the model using a dictionary to set values to the attributes of the class.
#     #     """
#     #     if attributesDict is None:
#     #         settings = {}
#     #         for k, v in cls.__dict__.items():
#     #             if not (callable(cls.__dict__[k])) and (k!='settings') and not (k.startswith('_')):
#     #                 # print(f'{k}: {v}')
#     #                 settings[k] = v
#     #         return(settings)
#     #     else:
#     #         for k, v in attributesDict.items():
#     #             setattr(cls, k, v)

#     def reset(self):
#         self._pattern = pd.DataFrame()

#     def updatepattern(self, prevCandle, candle):
#         if (prevCandle['High'] > candle['High']) and (prevCandle['Low'] < candle['Low']):
#             self._pattern = candle
#         else:
#             self.reset()

#         if self.debug:
#             print(f'{candle.name}: {self._ticker}')
        
#     def criteria(self, candle):
#         if self.selectCriteria is None:
#             return(True)
#         elif self.selectCriteria.lower() == '4sma':
#             if (candle['sma4'] > candle['sma17'] > candle['sma34'] > candle['sma80']):
#                 return(True)
#             else: return(False)
#         elif self.selectCriteria.lower()[:3] == 'sma':
#             if candle['Close'] > candle[self.selectCriteria.lower()]:
#                 return(True)
            
#     def checktrigger(self, candle):
#         """Checks if an entry is triggered on candle based on pattern.

#         Args:
#             candle (pd.Series): Time series pandas series
#         return (float | None, float | None): entry and stop
#         """
#         entry, stop = None, None
#         if not self._pattern.empty():
#             if candle['Low'] <= (self._pattern['High'] + 0.01) <= candle['High']:
#                 entry = round(self._pattern['High'] + 0.01, 2)
#                 stop = round(self._pattern['Low'] - 0.01, 2)
#         return(entry, stop)

#     def entrycallback(self, idx):
#         """
#         """
#         candle = self._trader._portfolio[self._ticker].iloc[idx]
#         prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1])
        
#         entry, stop = self.checktrigger(candle)
#         if entry and stop:
#             if self.criteria(prevCandle):
#                 # filter RR lower than 1:
#                 print(f'{candle.name}: {self._ticker} {entry=}, {stop=}')
#                 tradeID = self._trader.buy(ticker=self._ticker, entry=entry,
#                                     stop=stop, entry_date=candle.name, strategy_name=self.name)
#                 self._trader.settarget(tradeID, target=None)
#         self.updatepattern(prevCandle, candle)
    
#     # def searchcallback(self, idx):
#     #     """
#     #     """
#     #     candle = self._trader._portfolio[self._ticker].iloc[idx]
#     #     self.updatepattern(candle)

#     #     minC = self._pattern.loc[self._pattern['pivot']=='C', 'Low'].min()
        
#     #     if self.criteria(candle):
#     #         if self._pattern.iloc[-1]['pivot'] == 'S':
#     #             entry = round(self._pattern.iloc[-1]['High'] - 0.01, 2)
#     #             stop = round(min(minC, self._pattern.iloc[-1]['Low']) - 0.01, 2)
#     #             triggerOn = 'S'
#     #             # print(self._ticker, entry, stop)
#     #         elif self._pattern.iloc[-1]['pivot'] == 'C':
#     #             entry = round(self._pattern['High'].max() - 0.01, 2)
#     #             stop = round(minC - 0.01, 2)
#     #             triggerOn = 'C'
#     #             # print(self._ticker, entry, stop)
#     #         else:
#     #             entry, stop, triggerOn = None, None, None
#     #     else:
#     #         entry, stop, triggerOn = None, None, None
#     #     stop = None if stop is None else entry - self.stop['factor'] * (entry - stop)
#     #     # print(self._ticker, entry, stop)
#     #     return(entry, stop, triggerOn)
