import pandas as pd


class StopOrTarget:
    """AON
    """
    _ticker = 'MGLU3'
    _trader = {}  # trader object

    def __init__(self, ticker, trader):
        self._ticker = ticker
        self._trader = trader

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
                print('Profit!')

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
