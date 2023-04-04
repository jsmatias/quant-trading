import pandas as pd
# import numpy as np
from broker import Broker
from backtest import Backtest


class Trader():

    _initial_cap = 0
    _capital = 0
    _strategies = ()
    # broker = None
    # backtest = None

    _history = pd.DataFrame(columns=[
        'entry_datetime',
        'side',
        'ticker',
        'strategy',
        'entry',
        'entry_vol',
        'stop',
        'target',
        'status',
        'exit',
        'exit_vol',
        'exit_datetime',
    ])

    def __init__(self, capital=None):
        if capital:
            self.new(capital)
        self.broker = Broker(self)
        self.backtest = Backtest(self)
        
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

    @property
    def capital(self):
        """Returns the liquid capital
        """
        return(self._capital)
    
    @capital.setter
    def capital(self, cap):
        """Set the liquid capital 
        """
        self._capital = round(cap, 2)

    @property
    def strategies(self):
        return(self._strategies)

    @strategies.deleter
    def strategies(self):
        self._strategies = ()

    def addstrategy(self, strategy):
        """
        """
        strategy.bind(self)
        self._strategies += (strategy,)

    def new(self, capital):
        self._initial_cap = capital
        self.reset()

    def reset(self):
        """Resets the initial settings of the trader, such as capital, risk, etc. 
        """
        del self.strategies
        self.capital = self._initial_cap
        self._history = pd.DataFrame(columns=[
            'entry_datetime',
            'exit_datetime',
            'side',
            'strategy',
            'ticker',
            'entry_vol',
            'entry',
            'stop',
            'target',
            'exit',
            'exit_vol',
            'status',
            'fees'
        ])

    def isSold(self, ticker) -> pd.DataFrame:
        """Verifies if the trader has an open short position for a specific ticker"""
        
        _isSold = (self._history['ticker'] == ticker) & (self._history['side'] == 'short') & (self._history['status'] == 'open')
        return(_isSold)
    
    def isBought(self, ticker) -> pd.DataFrame:
        """Verifies if the trader has an open long position for a specific ticker"""
        
        _isBought = (self._history['ticker'] == ticker) & (self._history['side'] == 'long') & (self._history['status'] == 'open')
        return(_isBought)

    def buy(self, ticker, quantity, price, when, orderType, strategyName=None):
        """Registers a buying process and reduces that correct amount from capital.
        """
        price = self.broker.triggered(ticker, price, when, side='buy', orderType=orderType)
        if price:
            # First close any short position
            for idx, position in self._history[self.isSold(ticker)].iterrows():
                quantityToCover = abs(position['entry_vol'])
                # print('buy if sold: ', when, ' ', idx, ' ', quantityToCover, quantity)
                # print(self._history[self.isSold(ticker)])
                if quantity >= quantityToCover:
                    # close quantityToCover
                    self.close(idx, price, when, strategyName)
                    # update quantity
                    quantity -= quantityToCover
                elif quantity > 0:
                    # edit open position to quantity
                    positionCopy = position.copy()
                    self._history.loc[idx, 'entry_vol'] = - quantity
                    positionCopy['entry_vol'] = - (quantityToCover - quantity)
                    # close open position with quantity
                    self.close(idx, price, when, strategyName)
                    # update quantity
                    quantity -= quantity
                    # open a new short position same as previous with the remaining quantity (quantToCover - quantity)
                    self._history = self._history.append(positionCopy).reset_index(drop=True)

            # If no (more) shorts, buy the remaining quantity
            if quantity > 0:
                # enough capital to buy?
                if quantity * price < self.capital:
                    # buy long
                    fees = self.broker.calculatefees(price * quantity)
                    self._history = self._history.append(
                        pd.DataFrame({
                            'entry_datetime': [when],
                            'side': ['long'],
                            'strategy': [strategyName],
                            'ticker': [ticker],
                            'entry': [price],
                            'entry_vol': [quantity],
                            'status': ['open'],
                            'fees': [fees]
                        })
                    ).reset_index(drop=True)
                    self.capital -= price * quantity + fees
                    return(self._history.index[-1])
                else:
                    print('Low available capital...')
                    return(-1)
            elif quantity < 0:
                print('Error.. Quantity shouldnt be lower than 0')
                return(None)
            else: return(None) 
        else:
            return(None)

    def sell(self, ticker, quantity, price, when, orderType, strategyName=None):
        """Registers a selling process and adds that correct amount to capital.
        """
        price = self.broker.triggered(ticker, price, when, side='sell', orderType=orderType)
        if price:
            # First close any long position
            for idx, position in self._history[self.isBought(ticker)].iterrows():
                quantityToSell = abs(position['entry_vol'])
                # print('sell if bought: ', when, ' ', idx, ' ', quantityToSell)
                if quantity >= quantityToSell:
                    # close quantityToSell
                    self.close(idx, price, when, strategyName)
                    # update quantity
                    quantity -= quantityToSell
                elif quantity > 0:
                    # edit open position to quantity
                    positionCopy = position.copy()
                    self._history.loc[idx, 'entry_vol'] = quantity
                    positionCopy['entry_vol'] = quantityToSell - quantity
                    # close open position with quantity
                    self.close(idx, price, when, strategyName)
                    # update quantity
                    quantity -= quantity
                    # open a new short position same as previous with the remaining quantity (quantToCover - quantity)
                    self._history = self._history.append(positionCopy).reset_index(drop=True)

            # If no (more) longs, sell the remaining quantity
            if quantity > 0:
                # sell short
                fees = self.broker.calculatefees(price * quantity)
                self._history = self._history.append(
                    pd.DataFrame({
                        'entry_datetime': [when],
                        'side': ['short'],
                        'strategy': [strategyName],
                        'ticker': [ticker],
                        'entry': [price],
                        'entry_vol': [- quantity],
                        'status': ['open'],
                        'fees': [fees]
                    })
                ).reset_index(drop=True)
                
                self.capital += price * quantity - fees
                return(self._history.index[-1])
            elif quantity < 0:
                print('Error.. Quantity shouldnt be lower than 0')
                return(None)
            else: return(None) 
        else:
            return(None) 
    
    def close(self, pos_idx, price, when, strategyName=None):
        """Closes an open position"""
        price = round(price, 2)
        
        if self._history.loc[pos_idx, 'status'] == 'open':
            quantity = - self._history.loc[pos_idx, 'entry_vol']
            fees = self.broker.calculatefees(price * quantity)
            self._history.loc[pos_idx, [
                'strategy',
                'exit_datetime',
                'exit_vol',
                'exit',
                'status'
            ]] = [strategyName, when, quantity, price, 'closed']
            self._history.loc[pos_idx, 'fees'] += fees
            self.capital -= quantity * price + fees
            return(pos_idx)
        elif self._history.loc[pos_idx, 'status'] == 'closed':
            print(f'Position {pos_idx} already closed')
            return(-1)

    def setstop(self, pos_idx, stop):
        """Set stop loss.
        
        Args:
            pos_idx (int): index of the open trade on the trades history.
            stop (float): price at which the stop loss will be set. 
        """
        stop = round(stop, 2)
        if pos_idx >= 0:
            self._history.loc[pos_idx, 'stop'] = stop
        else: print(f"Couldn't find the trade {pos_idx} to set the stop to...")

    def settarget(self, pos_idx, target=None):
        """Set target price.
        
        Args:
            pos_idx (int): index of the open trade on the trades history.
            target (float): if None target will be calculated based on the risk size.
        """
        target = round(target, 2)
        if pos_idx >= 0:
            # entry, stop = self._history.loc[pos_idx, ['entry', 'stop']]
            # if target is None: print('Using trader target.')
            # target = target or ((self._trg / self._R) * (entry - stop) + entry)
            self._history.loc[pos_idx, 'target'] = target
        else: print(f"Couldn't find the trade {pos_idx} to set target to...")

    def history(self, ticker=None):
        if ticker:
            mask = self._history['ticker'] == ticker
            return(self._history[mask])
        else:
            return(self._history)
