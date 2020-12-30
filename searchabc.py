# import pandas_datareader as pdr
# from time import sleep
import pandas as pd
from datetime import datetime, timedelta
from source.trader import Trader
from source.entrymodels import ABC

fileName = './universe/Scanning.xlsx'
universe = pd.read_excel(fileName)

filters = (universe['Preço'] > 2.5) \
    & ~(universe['Sigla'].str.contains('34')) \
    & (universe['Vol (21d)'] / 21 > 100000) \
    & (universe['Vl Merc'] > 20000) \
    & (universe['365d'] > 0)

universe = universe[filters]

trader = Trader(risk_size=100, target_size=300, capital=20000, fees=5.5)

end = datetime.today()
end = datetime(day=7, month=12, year=2020)
start = end - timedelta(30)
trader.loadportfolio(universe['Sigla'].apply(lambda s: '.'.join([s, 'SA'])), start=start, end=end)

trader.search(entrymodel=ABC)
# trader.opportunities()