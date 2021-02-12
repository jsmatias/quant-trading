import streamlit as st
import pandas as pd
from source.trader import Trader
from source.entrymodels import ABC
from source.exitmodels import StopOrTarget
from datetime import date


ABC.settings({
    'selectCriteria': 'sma21'
})

stocklist = pd.read_excel('./universe/Scanning.xlsx')
summary = None

filters = (stocklist['Preço'] > 2.5) \
    & ~(stocklist['Sigla'].str.contains('34')) \
    & (stocklist['Vol (21d)'] / 21 > 100000) \
    & (stocklist['Vl Merc'] > 20000)
stocklist = stocklist[filters]
# st.table(stocklist['Sigla'])

st.sidebar.title('QT-lab')

trader_expander = st.sidebar.beta_expander(label='Risk management')
portfolio_expander = st.sidebar.beta_expander(label='Portfolio')

risk = trader_expander.number_input('Risk size', 10.0, value=10.0, step=10.0, format='%.2f')
target = trader_expander.number_input('Target size', 10.0, value=10.0, step=10.0, format='%.2f')
capital = trader_expander.number_input('Initial capital', 1000.0, value=10000.0, step=100.0, format='%.2f')
fees =  trader_expander.number_input('Fees per trade', 0.0, value=5.0, step=0.01, format='%.2f')


ticker = portfolio_expander.selectbox('Pick a ticker', list(stocklist['Sigla'].sort_values())) + '.SA'
startDate = portfolio_expander.date_input('Start period', value=date(2010, 1, 1), min_value=date(2010, 1, 1), max_value=date.today())
endDate = portfolio_expander.date_input('End period', value=startDate, min_value=startDate, max_value=date.today())

runBacktest = st.sidebar.button('Run backtest')
if runBacktest:
    trader = Trader(risk_size=risk, target_size=target, capital=capital, fees=fees)
    trader.loadportfolio([ticker], start=startDate, end=endDate)
    trader.backtest(ABC, StopOrTarget, updaterisk=False, trail=False)
    st.table(trader.summary(ticker))
    st.plotly_chart(trader.plottrades(ticker))

