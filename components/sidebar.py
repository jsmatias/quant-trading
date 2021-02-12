import streamlit as st
from datetime import date
# from source.trader import Trader
from source.entrymodels import ABC
from source.exitmodels import StopOrTarget

def render(trader):

    riskExpand = st.sidebar.beta_expander('Risk management')
    strategyExpand = st.sidebar.beta_expander('Strategy')
    portfolioExpand = st.sidebar.beta_expander('Portfolio')

    risk = riskExpand.number_input('Risk size', 10.0, value=10.0, step=10.0, format='%.2f')
    target = riskExpand.number_input('Target size', 10.0, value=10.0, step=10.0, format='%.2f')
    capital = riskExpand.number_input('Initial capital', 1000.0, value=10000.0, step=100.0, format='%.2f')
    fees =  riskExpand.number_input('Fees per trade', 0.0, value=5.0, step=0.01, format='%.2f')

    ticker = portfolioExpand.selectbox('Pick a ticker', ['VALE3', 'LWSA3']) + '.SA'
    startDate = portfolioExpand.date_input('Start period', value=date(2010, 1, 1), min_value=date(2010, 1, 1), max_value=date.today())
    endDate = portfolioExpand.date_input('End period', value=startDate, min_value=startDate, max_value=date.today())

    action = st.sidebar.button('Run')
    if action:
        with st.spinner('Running...'):
            trader.new(risk_size=risk, target_size=target, capital=capital, fees=fees)
            trader.loadportfolio([ticker], start=startDate, end=endDate)
            trader.backtest(ABC, StopOrTarget, updaterisk=False, trail=False)
        st.success('Done!')
    
    return(trader)
