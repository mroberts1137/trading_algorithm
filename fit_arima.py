import os
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import yfinance as yf


def fit_arima(data, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    tmp_mdl = sm.tsa.arima.ARIMA(data, order=(p, d, q)).fit()
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (p, d, q)
                        best_model = tmp_mdl
                except:
                    continue

    # print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

    return best_model, best_order, best_aic


if __name__ == '__main__':
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2024, 1, 1)

    symbols = ['SPY', 'TLT', 'MSFT']

    msft = yf.Ticker("MSFT")

    # get historical market data
    df = msft.history(period="1mo")

    # print(df.head(30))
    data = df['Close'].values
    print(data)

    # data = pd.DataFrame({sym: get_px(sym) for sym in symbols})

    # log returns
    # lrets = np.log(data / data.shift(1)).dropna()

    # Notice I’ve selected a specific time period to run this analysis
    # TS = lrets.SPY.loc['2020':'2023']
    res_tup = get_best_model(data)

    print(res_tup.best_mdl.summary())
