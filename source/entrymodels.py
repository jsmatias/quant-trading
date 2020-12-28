import pandas as pd


class ABC01:
    """ABC with entry above C and stop bellow A
    ABC on close.
    Rules:
        + Entry above signal after C.High or above B.High
        + Stop bellow C.Low
        + Target:  
    """

    _A = pd.Series(dtype='float64')
    _B = pd.Series(dtype='float64')
    _C = pd.Series(dtype='float64')
    _ticker = 'MGLU3'
    _trader = {}  # trader object

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])

    def resetABC(self, initialCandle):
        self._A = initialCandle
        self._B = pd.Series(dtype='float64')
        self._C = pd.Series(dtype='float64')

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C = candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._C['High']:
                # triggered
                entry = round(self._C['High'], 2)
                stop = round(self._A['Low'], 2)
                print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                self._trader.buy(ticker=self._ticker, entry=entry,
                                 stop=stop, entry_date=candle.name)

                self.resetABC(initialCandle=self._C)


class ABC03:
    """ABC on close.
    Rules:
        + Entry above signal (S.High) after C or above B.High
        + Stop bellow C.Low
        + Target fixed according to RR 
    """

    _A = pd.Series(dtype='float64')
    _B = pd.Series(dtype='float64')
    _C = pd.Series(dtype='float64')
    _S = pd.Series(dtype='float64')
    _ticker = 'MGLU3'
    _trader = {}  # trader object

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])

    def resetABC(self, initialCandle):
        self._A = initialCandle
        self._B = pd.Series(dtype='float64')
        self._C = pd.Series(dtype='float64')
        self._S = pd.Series(dtype='float64')

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]
        entry = None
        stop = None

        if (candle['Close'] <= prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
                print('set a')
            elif candle['Low'] > self._A['Low']:
                print('set c, reset s')

                self._C = candle
                self._S = pd.Series(dtype='float64')

            else:
                # reset
                self.resetABC(candle)
                print('reset, down')

        else:
            # uptrend
            if self._C.empty:
                self._B = candle
                print('set b')


            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._B['High']):
                # reset
                self.resetABC(candle)
                print('reset, up')


            elif candle['High'] > max(self._B['High'], self._C['High']) :
                # triggered
                print('trig on b')

                entry = round(max(self._B['High'], self._C['High']), 2) + 0.01
                stop = round(self._C['Low'], 2) - 0.01

            elif self._S.empty:
                print('set s')

                # set signal bar
                self._S = candle

            elif candle['High'] > self._S['High']:
                # triggered
                print('trig on s')

                entry = round(self._S['High'], 2) + 0.01
                stop = round(min(self._C['Low'], self._S['Low']), 2) - 0.01

            input('> ')    
                
        if bool(entry) and bool(stop):
            print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

            tradeID = self._trader.buy(ticker=self._ticker, entry=entry,
                                stop=stop, entry_date=candle.name)
            self._trader.settarget(tradeID)

            self.resetABC(initialCandle=self._C)


class ABC04:
    """ABC on close.
    Rules:
        + Entry above signal (S.High) after C or above B.High
        + Stop bellow C.Low
        + Target fixed according to RR 
    """
    pattern = pd.DataFrame()
    debug = False
    useTarget = False
    selectCriteria = None # 4SMA, SMA21...
    stop = {'position': 'C', 'factor': 1}

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

    def reset(self, initialCandle):
        self.pattern = pd.DataFrame(initialCandle).transpose()
        self.pattern['pivot'] = 'A'

    def add(self, pivot, candle):
        candle = candle.copy()
        candle['pivot'] = pivot
        self.pattern = self.pattern.append(candle)

    def updatepattern(self, candle):
        
        prevCandle = self.pattern.iloc[-1]
        if prevCandle['pivot'] == 'A':
            if (candle['Close'] <= prevCandle['Close']):
                self.reset(candle)
                msg = 'a down -> a'
            else:
                self.add('B', candle)
                msg = 'a up -> b'
        elif prevCandle['pivot'] == 'B':
            if (candle['Close'] >= prevCandle['Close']) or ((candle['Close'] < prevCandle['Close']) and (candle['High'] > prevCandle['High'])):
                self.add('B', candle)
                msg = 'b up -> b'
            elif (candle['Low'] >= min(self.pattern['Low'])):
                self.add('C', candle)
                msg = 'b down -> c'
            else:
                self.reset(candle)
                msg = 'b min -> a'
        elif prevCandle['pivot'] == 'C':
            if (candle['Low'] < min(self.pattern['Low'])):
                self.reset(prevCandle)
                msg = 'c min -> a'
            elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] <= max(self.pattern['High'])):
                self.add('S', candle)
                msg = 'c up -> s'
            elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] > max(self.pattern['High'])):
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
            if (candle['Close'] >= prevCandle['Close']):
                # S is B
                self.reset(self.pattern.iloc[-2])
                self.add('B', prevCandle)
                self.add('B', candle)
                msg = 's up -> a,b,b'
            elif (candle['Low'] >= min(self.pattern['Low'])) and (candle['High'] <= prevCandle['High']):
                self.add('C', candle)
                msg = 's down -> c'
            else:
                self.reset(candle)
                msg = 's down -> a'
        
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
        if self.pattern.iloc[-1]['pivot'] == 'S':
            minC = min(self.pattern.loc[self.pattern['pivot']=='C', 'Low'])
            tradingRange = {
                'min': min(self.pattern.iloc[-1]['Low'], minC),
                'max1': self.pattern.iloc[-1]['High'], # S high or max of S and C?
                'max2': max(self.pattern['High'])
            }
            if self.stop['position'] == 'C':
                stop = round(tradingRange['min'] - 0.01, 2)
            elif self.stop['position'] == 'A':
                stop = round(self.pattern['Low'].min() - 0.01, 2)
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
                
        if self.pattern.iloc[-1]['pivot'] == 'C':
            tradingRange = {
                'min': min(self.pattern.loc[self.pattern['pivot']=='C', 'Low']),
                'max': max(self.pattern['High'])
            }
            if self.stop['position'] == 'C':
                stop = round(tradingRange['min'] - 0.01, 2)
            elif self.stop['position'] == 'A':
                stop = round(self.pattern['Low'].min() - 0.01, 2)
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
                print(f'{candle.name}: {self._ticker} {entry=}, {stop=}')
                tradeID = self._trader.buy(ticker=self._ticker, entry=entry,
                                    stop=stop, entry_date=candle.name)
                target = (self._B['High'] - self._A['Low']) + entry if self.useTarget else 0
                self._trader.settarget(tradeID, target=target)

        self.updatepattern(candle)


class ABC04_4SMA(ABC04):
    """ABC on close.
    Rules:
        + select strong trends by SMA4, SMA17, SMA34, SMA80
        + Entry above signal (S.High) after C or above B.High
        + Stop bellow C.Low
        + Target fixed according to RR 
    """

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.reset(trader._portfolio[self._ticker].iloc[0])
        self._trader._portfolio[ticker]['sma4'] = self._trader._portfolio[ticker]['Close'].rolling(window=4).mean()
        self._trader._portfolio[ticker]['sma17'] = self._trader._portfolio[ticker]['Close'].rolling(window=17).mean()
        self._trader._portfolio[ticker]['sma34'] = self._trader._portfolio[ticker]['Close'].rolling(window=34).mean()
        self._trader._portfolio[ticker]['sma80'] = self._trader._portfolio[ticker]['Close'].rolling(window=80).mean()


    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]
        entry = None
        stop = None

        # check entries
        if self.pattern.iloc[-1]['pivot'] == 'S':
            minC = min(self.pattern.loc[self.pattern['pivot']=='C', 'Low'])
            tradingRange = {
                'min': min(self.pattern.iloc[-1]['Low'], minC),
                'max1': self.pattern.iloc[-1]['High'], # S high or max of S and C?
                'max2': max(self.pattern['High'])
            }
            stop = round(tradingRange['min'] - 0.01, 2)
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
            # if triggers on C
            elif (
                (tradingRange['max1'] + 0.01 < candle['Open'] <= tradingRange['max2']) \
                and (candle['High'] > tradingRange['max2'])
            ) or (
                (candle['Open'] > tradingRange['max2']) \
                and (candle['Low'] <= tradingRange['max2'] + 0.01) 
            ):
                entry = round(tradingRange['max2'] + 0.01, 2)

        if self.pattern.iloc[-1]['pivot'] == 'C':
            tradingRange = {
                'min': min(self.pattern.loc[self.pattern['pivot']=='C', 'Low']),
                'max': max(self.pattern['High'])
            }
            stop = round(tradingRange['min'] - 0.01, 2)
            if ((tradingRange['min'] <= candle['Open']) \
                and (tradingRange['max'] <= candle['High'] + 0.01) and (candle['Low'] <= tradingRange['max'] + 0.01)):
                entry = round(tradingRange['max'] + 0.01, 2)

        if bool(entry) and bool(stop):
            if (candle['sma4'] > candle['sma17'] > candle['sma34'] > candle['sma80']):
                print(f'{candle.name}: {self._ticker} {entry=}, {stop=}')
                tradeID = self._trader.buy(ticker=self._ticker, entry=entry,
                                    stop=stop, entry_date=candle.name)
                target = (self._B['High'] - self._A['Low']) + entry if self.useTarget else 0
                self._trader.settarget(tradeID, target=target)

        # pattern
        if self.pattern.iloc[-1]['pivot'] == 'A':
            if (candle['Close'] <= prevCandle['Close']):
                self.reset(candle)
                msg = 'a down -> a'
            else:
                self.add('B', candle)
                msg = 'a up -> b'
        elif self.pattern.iloc[-1]['pivot'] == 'B':
            if (candle['Close'] >= prevCandle['Close']) or ((candle['Close'] < prevCandle['Close']) and (candle['High'] > prevCandle['High'])):
                self.add('B', candle)
                msg = 'b up -> b'
            elif (candle['Low'] >= min(self.pattern['Low'])):
                self.add('C', candle)
                msg = 'b down -> c'
            else:
                self.reset(candle)
                msg = 'b min -> a'
        elif self.pattern.iloc[-1]['pivot'] == 'C':
            if (candle['Low'] < min(self.pattern['Low'])):
                self.reset(prevCandle)
                msg = 'c min -> a'
            elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] <= max(self.pattern['High'])):
                self.add('S', candle)
                msg = 'c up -> s'
            elif (candle['Close'] >= prevCandle['Close']) and (candle['High'] > max(self.pattern['High'])):
                # trigger on B high
                self.reset(prevCandle)
                self.add('B', candle)
                msg = 'c up -> trigger on b'
            elif (candle['Close'] < prevCandle['Close']):
                self.add('C', candle)
                msg = 'c down -> c'
            else:
                print('Condition not taken into account')
        elif self.pattern.iloc[-1]['pivot'] == 'S':
            if (candle['Close'] >= prevCandle['Close']):
                # S is B
                self.reset(self.pattern.iloc[-2])
                self.add('B', prevCandle)
                self.add('B', candle)
                msg = 's up -> a,b,b'
            elif (candle['Low'] >= min(self.pattern['Low'])) and (candle['High'] <= prevCandle['High']):
                self.add('C', candle)
                msg = 's down -> c'
            else:
                self.reset(candle)
                msg = 's down -> a'

        if self.debug:
            print(f'{candle.name}: {self._ticker} {msg}')


class ABC_SMA_CROSS01(ABC01):

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])
        self._trader._portfolio[ticker]['sma9'] = self._trader._portfolio[ticker]['Close'].rolling(window=9).mean()
        self._trader._portfolio[ticker]['sma21'] = self._trader._portfolio[ticker]['Close'].rolling(window=21).mean()

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C =  candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._C['High']:
                if candle['sma9'] > candle['sma21']:
                    # triggered
                    entry = round(self._C['High'], 2)
                    stop = round(self._A['Low'], 2)
                    print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                    self._trader.buy(ticker=self._ticker, entry=entry,
                                 stop=stop, entry_date=self._C.name)

                self.resetABC(initialCandle=self._C)


class ABC_SMA01(ABC01):
    """ABC with C above SMA21
    """


    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])
        # self._trader._portfolio[ticker]['sma9'] = self._trader._portfolio[ticker]['Close'].rolling(window=9).mean()
        self._trader._portfolio[ticker]['sma21'] = self._trader._portfolio[ticker]['Close'].rolling(window=21).mean()

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C =  candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._C['High']:
                if self._C['Low'] > candle['sma21']:
                    # triggered
                    entry = round(self._C['High'], 2)
                    stop = round(self._A['Low'], 2)
                    print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                    self._trader.buy(ticker=self._ticker, entry=entry,
                                 stop=stop, entry_date=self._C.name)

                self.resetABC(initialCandle=self._C)


class ABC_SMA03(ABC01):
    """ABC with C above SMA21 and no resistance on previous 6 months
    Excellence criteria
    """


    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])
        # self._trader._portfolio[ticker]['sma9'] = self._trader._portfolio[ticker]['Close'].rolling(window=9).mean()
        self._trader._portfolio[ticker]['sma21'] = self._trader._portfolio[ticker]['Close'].rolling(window=21).mean()

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C =  candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._C['High']:
                # candle above SMA21
                # if self._C['Low'] > candle['sma21']:
                if 1:
                # if share.iloc[-1]['Close'] > share.iloc[-1]['sma21']:
                    # Max of the month above the last 6 months
                    share = self._trader._portfolio[self._ticker].iloc[idx-3*21:idx]
                    # It can be adapt to no high above B[High]
                    # nAboveMax = len(share[share['High'] > share.iloc[-21:]['High'].max()]) 
                    nAboveMax = len(share.iloc[:-1][share.iloc[:-1]['High'] > self._B['High']]) 
                    if nAboveMax == 0:
                        # 60% of time price above sma21
                        nAboveSma = len(share[share['Low'] >= share['sma21']])
                        if nAboveSma > 0.5 * len(share):

                    # # no important resistance on previous 6 months
                    # share = self._trader._portfolio[self._ticker].iloc[idx-6*21:idx]

                    # nAboveMax = len(share[share['High'] > share.iloc[-21:]['High'].max()])
                    # if nAboveMax == 0:
                        # triggered
                            entry = round(self._C['High'], 2)
                            stop = round(self._A['Low'], 2)
                            print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                            self._trader.buy(ticker=self._ticker, entry=entry,
                                    stop=stop, entry_date=self._C.name)

                self.resetABC(initialCandle=self._C)


class ABC_SMA02(ABC01):
    """ABC with C above SMA21 and stop below SMA21
    """


    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])
        # self._trader._portfolio[ticker]['sma9'] = self._trader._portfolio[ticker]['Close'].rolling(window=9).mean()
        self._trader._portfolio[ticker]['sma21'] = self._trader._portfolio[ticker]['Close'].rolling(window=21).mean()

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C =  candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._C['High']:
                if self._C['Low'] > candle['sma21']:
                    # triggered
                    entry = round(self._C['High'], 2)
                    stop = round(candle['sma21'] - 0.01, 2)
                    print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                    self._trader.buy(ticker=self._ticker, entry=entry,
                                 stop=stop, entry_date=self._C.name)

                self.resetABC(initialCandle=self._C)


class ABC02(ABC01):
    """ABC with entry above B and stop bellow C
    """

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C = candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._B['High']:
                # triggered
                entry = round(self._B['High'], 2)
                stop = round(self._C['Low'], 2)
                print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                self._trader.buy(ticker=self._ticker, entry=entry,
                                 stop=stop, entry_date=self._C.name)

                self.resetABC(initialCandle=self._C)


class ABC_SMA_CROSS02(ABC01):

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader
        self.resetABC(trader._portfolio[self._ticker].iloc[0])
        self._trader._portfolio[ticker]['sma9'] = self._trader._portfolio[ticker]['Close'].rolling(window=9).mean()
        self._trader._portfolio[ticker]['sma21'] = self._trader._portfolio[ticker]['Close'].rolling(window=21).mean()

    def entrycallback(self, idx):

        candle = self._trader._portfolio[self._ticker].iloc[idx]
        prevCandle = self._trader._portfolio[self._ticker].iloc[idx - 1]

        if (candle['Close'] < prevCandle['Close']):
            # downtrend
            if self._B.empty:
                self._A = candle
            elif candle['Low'] > self._A['Low']:
                self._C =  candle
            else:
                # reset
                self.resetABC(candle)
        else:
            # uptrend
            if self._C.empty:
                self._B = candle

            # A, B and C set -> check if it triggers or reset
            elif (candle['Low'] < self._A['Low']) or (candle['Low'] > self._C['High']):
                # reset
                self.resetABC(candle)

            elif candle['High'] > self._C['High']:
                if candle['sma9'] > candle['sma21']:
                    # triggered
                    entry = round(self._B['High'], 2)
                    stop = round(self._C['Low'], 2)
                    print(f'{self._C.name}: {self._ticker} {entry=}, {stop=}')

                    self._trader.buy(ticker=self._ticker, entry=entry,
                                 stop=stop, entry_date=self._C.name)

                self.resetABC(initialCandle=self._C)


