import pandas as pd
import numpy as np

# Improvement: add method to calculate stop and target
# Improvement: queue orders to be triggered or not. 
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


class RSIN:
    """RSI-N: Relative strength index with N periods
    Rules:
        + Entry on close of the bar when RSI-N < threshold
        + Stop: on time, 7 bars with no profit, or predefined
        + Target max of the high of previous 2 bars

        + filter: above SMA-P, simple moving avarage of P periods for bull market
    """

    debug = False
    stop = {'type': None, 'factor': 1.3} # type: 'time' or fixed 
    target = None # 'AB' or None 
    selectCriteria = 'SMA80' # 4SMA, SMA21...
    name = 'RSI-N'
    rsiLimit = 10
    N = 2
    # divideCapital = [1]

    _ticker = 'B3SA3'
    _trader = {}  # trader object

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        
        # RSI calculation
        isGain = self._trader.portfolio(self._ticker)['Close'].diff(1) > 0
        isLoss = self._trader.portfolio(self._ticker)['Close'].diff(1) < 0

        self._trader.portfolio(self._ticker)['diff'] = self._trader.portfolio(self._ticker)['Close'].diff(1)
        self._trader.portfolio(self._ticker)['gain'] = self._trader.portfolio(self._ticker).loc[isGain, 'diff']
        self._trader.portfolio(self._ticker)['loss'] = self._trader.portfolio(self._ticker).loc[isLoss, 'diff']
        self._trader.portfolio(self._ticker)['gain'].fillna(0, inplace=True)
        self._trader.portfolio(self._ticker)['loss'].fillna(0, inplace=True)

        avgLoss = (self.N - 1) * [np.nan] + [- self._trader.portfolio(self._ticker)['loss'][:self.N].mean()]
        avgGain = (self.N - 1) * [np.nan] + [self._trader.portfolio(self._ticker)['gain'][:self.N].mean()]
        for _, candle in self._trader.portfolio(self._ticker).iloc[2:].iterrows():
            avgGain.append((avgGain[-1] * (self.N - 1) + candle['gain']) / self.N)
            avgLoss.append((avgLoss[-1] * (self.N - 1) - candle['loss']) / self.N)
            
        avgGain = np.array(avgGain)
        avgLoss = np.array(avgLoss)

        rsi = 100 - 100 / (1 + avgGain / avgLoss)
        self._trader.portfolio(self._ticker)[f'rsi{self.N}'] = rsi
        self._trader.portfolio(self._ticker).drop(columns=['diff', 'gain', 'loss', ], inplace=True)

        if self.selectCriteria is None:
            pass
        # elif self.selectCriteria.lower() == '4sma':
        #     self._trader._portfolio[ticker]['sma4'] = self._trader._portfolio[ticker]['Close'].rolling(window=4).mean()
        #     self._trader._portfolio[ticker]['sma17'] = self._trader._portfolio[ticker]['Close'].rolling(window=17).mean()
        #     self._trader._portfolio[ticker]['sma34'] = self._trader._portfolio[ticker]['Close'].rolling(window=34).mean()
        #     self._trader._portfolio[ticker]['sma80'] = self._trader._portfolio[ticker]['Close'].rolling(window=80).mean()
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

    # def reset(self):
    #     self._pattern = pd.DataFrame()

    # def updatepattern(self, prevCandle, candle):
    #     if (prevCandle['High'] > candle['High']) and (prevCandle['Low'] < candle['Low']):
    #         self._pattern = candle
    #     else:
    #         self.reset()

    #     if self.debug:
    #         print(f'{candle.name}: {self._ticker}')
        
    def criteria(self, candle):
        if self.selectCriteria is None:
            return(True)
        # elif self.selectCriteria.lower() == '4sma':
        #     if (candle['sma4'] > candle['sma17'] > candle['sma34'] > candle['sma80']):
        #         return(True)
        #     else: return(False)
        elif self.selectCriteria.lower()[:3] == 'sma':
            if candle['Close'] > candle[self.selectCriteria.lower()]:
                return(True)
        else:
            print('Criteria not available, returning True!')
            return(True)
            
    def buildsetup(self, candle, prevCandle):
        """Checks if an entry is triggered on candle based on pattern.

        Args:
            candle (pd.Series): Time series pandas series
        return (float | None, float | None): entry and stop
        """
        entry, stop, target, vol = None, None, None, None
        if candle[f'rsi{self.N}'] < self.rsiLimit:
            entry = round(candle['Close'], 2)
            # TODO: implement the other types of stop loss
            stop = 0.01 if self.stop['type'] is None else round(
                candle['Low'] - (candle['High'] - candle['Low']) * self.stop['factor'], 2
            )
            target = round(max([candle['High'], prevCandle['High']]), 2)
            vol = int(self._trader._capital / entry // 100 * 100)
        return(entry, stop, target, vol)

    def entrycallback(self, idx):
        """
        """
        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]
        
        entry, stop, target, vol = self.buildsetup(candle, prevCandle)
        if entry and stop and target and vol:
            if self.criteria(candle):
                # filter RR lower than 1:
                print(f'{candle.name}: {self._ticker} {entry=}, {stop=}, {target=}, {vol=}')
                tradeID = self._trader.buy(ticker=self._ticker, entry=entry,
                                    stop=stop, entry_date=candle.name, strategy_name=self.name, vol=vol)
                self._trader.settarget(tradeID, target=target)
        # self.updatepattern(prevCandle, candle)
    
    def exitcallback(self, dayIdx):
        """
        """
        candle = self._trader._portfolio[self._ticker].iloc[dayIdx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[dayIdx - 1]
        
        boughtFilter = (self._trader._history['status'] == 'bought') & (
            self._trader._history['ticker'] == self._ticker)
        for idx, position in self._trader._history[boughtFilter].iterrows():
            price = None
            atDate = None
            if self._trader._history.loc[idx, 'entry_date'] < candle.name:
                # Consider gaps
                if candle['High'] >= position['target']:
                    if candle['Open'] >= position['target']:
                        price = round(candle['Open'], 2)
                    else:
                        price = round(position['target'], 2)

                    atDate = candle.name
                    print('Profit!')

                elif candle['Low'] <= position['stop']:
                    if candle['Open'] <= position['stop']:
                        price = round(candle['Open'], 2)
                    else:
                        price = round(position['stop'], 2)
                    atDate = candle.name
                    print('Loss...')
                else:
                    # update target
                    target = round(max([candle['High'], prevCandle['High']]), 2)
                    self._trader.settarget(idx, target=target)

                if price and atDate:
                    self._trader.sell(price, atDate, idx)
                    print(f'{atDate}: {self._ticker} {price=}')
            else: pass
    # def searchcallback(self, idx):
    #     """
    #     """
    #     candle = self._trader._portfolio[self._ticker].iloc[idx]
    #     self.updatepattern(candle)

    #     minC = self._pattern.loc[self._pattern['pivot']=='C', 'Low'].min()
        
    #     if self.criteria(candle):
    #         if self._pattern.iloc[-1]['pivot'] == 'S':
    #             entry = round(self._pattern.iloc[-1]['High'] - 0.01, 2)
    #             stop = round(min(minC, self._pattern.iloc[-1]['Low']) - 0.01, 2)
    #             triggerOn = 'S'
    #             # print(self._ticker, entry, stop)
    #         elif self._pattern.iloc[-1]['pivot'] == 'C':
    #             entry = round(self._pattern['High'].max() - 0.01, 2)
    #             stop = round(minC - 0.01, 2)
    #             triggerOn = 'C'
    #             # print(self._ticker, entry, stop)
    #         else:
    #             entry, stop, triggerOn = None, None, None
    #     else:
    #         entry, stop, triggerOn = None, None, None
    #     stop = None if stop is None else entry - self.stop['factor'] * (entry - stop)
    #     # print(self._ticker, entry, stop)
    #     return(entry, stop, triggerOn)


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
