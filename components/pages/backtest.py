import streamlit as st


def render(trader):

    st.header('Backtest')
    
    if trader.history().empty:
        st.write('Select your strategy and equity to perform a backtest.')
        st.write('Your results will be shown here.')
    else:
        ticker = st.selectbox('Select the ticker to see a specific result', list(trader._portfolio.keys()))
        summaryExpand = st.beta_expander(label='Show Trade Summary')
        performanceExpand = st.beta_expander(label='Show Backtest Performance')
        graphExpand = st.beta_expander(label='Show Trades On The Graph')
        historyExpand = st.beta_expander(label='Show Trade History')

        summaryExpand.table(trader.summary(ticker))
        performanceExpand.table(trader.performance(ticker))
        graphExpand.plotly_chart(trader.plottrades(ticker))
        historyExpand.dataframe(trader.history(ticker))
