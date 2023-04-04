from datetime import time, date, datetime
import plotly.graph_objects as go


def strtodatetime(strdatetime):
    """ dd-mm-YYYY HH:MM
    """
    if type(strdatetime) is datetime:
        return(strdatetime)
    else:
        dt, t = strdatetime.split(' ')
        day, month, year = (int(n) for n in dt.split('-'))
        hour, minute = (int(n) for n in t.split(':'))
        return(datetime(year, month, day, hour, minute))


def strtodate(strdate):
    """ dd-mm-YYYY
    """
    if type(strdate) is date:
        return(strdate)
    else:
        day, month, year = (int(n) for n in strdate.split('-'))
        return(date(year, month, day))


def strtotime(strtime):
    """ HH:MM
    """
    if type(strtime) is time:
        return(strtime)
    else:
        hour, minute = (int(n) for n in strtime.split(':'))
        return(time(hour, minute))


def maketradesplot(timeseries, axis='y') -> list:
    """Makes the plots for long and short trades"""

    buyColor = 'darkturquoise'
    sellColor = 'blueviolet'

    long = timeseries['side'] == 'long'
    short = timeseries['side'] == 'short'
    longEntries = go.Scatter(
        x=timeseries[long]['entry_datetime'],
        y=timeseries[long]['entry'],
        name=f'Buy long',
        text=timeseries[long]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
        yaxis=axis,
        mode='markers',
        marker=dict(color=buyColor, symbol='triangle-up')
    )

    longExits = go.Scatter(
        x=timeseries[long]['exit_datetime'],
        y=timeseries[long]['exit'],
        name=f'Sell',
        text=timeseries[long]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
        yaxis=axis,
        mode='markers',
        marker=dict(color=sellColor, symbol='triangle-down')
    )

    shortEntries = go.Scatter(
        x=timeseries[short]['entry_datetime'],
        y=timeseries[short]['entry'],
        name=f'Sell short',
        text=timeseries[short]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
        yaxis=axis,
        mode='markers',
        marker=dict(color=sellColor, symbol='triangle-up')
    )

    shortExits = go.Scatter(
        x=timeseries[short]['exit_datetime'],
        y=timeseries[short]['exit'],
        name=f'Buy to cover',
        text=timeseries[short]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
        yaxis=axis,
        mode='markers',
        marker=dict(color=buyColor, symbol='triangle-down')
    )

    return([longEntries, longExits, shortEntries, shortExits])


def makecandlesticks(timeseries, ticker='Time series', axis='y', colors={'increasing': 'white', 'decreasing': 'black'}):

    candleSticksChart = go.Candlestick(
        x=timeseries.index,
        open=timeseries['Open'], high=timeseries['High'],
        low=timeseries['Low'], close=timeseries['Close'],
        increasing_line_color=colors['increasing'], decreasing_line_color=colors['decreasing'],
        name=ticker,
        yaxis=axis
    )
    return(candleSticksChart)


def maketopsandbottoms(timeseries, axis='y') -> list:

    topMask = (timeseries['High'] > timeseries['High'].shift(1)) & \
        (timeseries['High'] > timeseries['High'].shift(-1))
    bottomMask = (timeseries['Low'] < timeseries['Low'].shift(1)) & \
        (timeseries['Low'] < timeseries['Low'].shift(-1))
    timeseries.loc[topMask, 'Top'] = timeseries.loc[topMask, 'High']
    timeseries.loc[bottomMask, 'Bottom'] = timeseries.loc[bottomMask, 'Low']

    tops = go.Scatter(
        x=timeseries.index,
        y=timeseries['Top'],
        name='Tops',
        yaxis=axis,
        mode='markers',
        marker=dict(symbol='arrow-bar-down', color='#46039f',
                    line=dict(width=1, color='#0d0887'))
        # marker=dict(symbol='y-up', line=dict(width=2, color='orange'))
    )

    bottoms = go.Scatter(
        x=timeseries.index,
        y=timeseries['Bottom'],
        name='Bottoms',
        yaxis=axis,
        mode='markers',
        marker=dict(symbol='arrow-bar-up', color='#46039f',
                    line=dict(width=1, color='#0d0887'))
    )

    return([tops, bottoms])


def makeindicatorplot(timeseries, indicator):

    indicatorPlot = go.Scatter(
        x=timeseries.index,
        y=timeseries[indicator['name']],
        name=indicator['name'].upper(),
        line=dict(color=indicator['color']),
        yaxis=indicator['axis']
    )
    # go.Bar(x=self._trader.broker.portfolio[ticker].index, y=self._trader.broker.portfolio[ticker]['Volume'], name='Volume', yaxis='y')
    return(indicatorPlot)


# def maketradesplot(timeseries, axis='y') -> list:
#     """Makes the plots for long and short trades"""

#     long = timeseries['side'] == 'long'
#     short = timeseries['side'] == 'short'
#     longEntries = go.Scatter(
#         x=timeseries[long]['entry_datetime'],
#         y=timeseries[long]['entry'],
#         name=f'Buy long',
#         text=timeseries[long]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
#         yaxis=axis,
#         mode='markers',
#         marker=dict(color='seagreen', symbol='triangle-up')
#     )

#     longExits = go.Scatter(
#         x=timeseries[long]['exit_datetime'],
#         y=timeseries[long]['exit'],
#         name=f'Sell',
#         text=timeseries[long]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
#         yaxis=axis,
#         mode='markers',
#         marker=dict(color='crimson', symbol='triangle-down')
#     )

#     shortEntries = go.Scatter(
#         x=timeseries[short]['entry_datetime'],
#         y=timeseries[short]['entry'],
#         name=f'Sell short',
#         text=timeseries[short]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
#         yaxis=axis,
#         mode='markers',
#         marker=dict(color='crimson', symbol='triangle-up')
#     )

#     shortExits = go.Scatter(
#         x=timeseries[short]['exit_datetime'],
#         y=timeseries[short]['exit'],
#         name=f'Buy to cover',
#         text=timeseries[short]['pnl'].apply(lambda x: f"PNL: {x:.2f}"),
#         yaxis=axis,
#         mode='markers',
#         marker=dict(color='seagreen', symbol='triangle-down')
#     )

#     return([longEntries, longExits, shortEntries, shortExits])
