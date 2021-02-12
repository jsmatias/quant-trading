import streamlit as st
import pandas as pd
from components.pages import home, backtest
from components import sidebar
from source.traderobj import trader


pages = {'Home': home, 'Backtest': backtest}
# @st.cache
# def start():
#     global trader
# trader = Trader()

# start()

sidebarHeader = st.sidebar
with sidebarHeader:
    st.title('Quant-trading lab')
    page = st.selectbox('Go to', options=list(pages.keys()))

trader = sidebar.render(trader)
pages[page].render(trader)


