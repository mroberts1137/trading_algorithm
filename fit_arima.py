import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels as sm


def get_px(x):
    return web.DataReader(x, 'yahoo', start=start, end=end)['Adj Close']


def _get_best_model(ts):
    best_aic = np.inf
    best_order = None
    best_mdl = None
    pq_rng = range(5)  # [0,1,2,3,4]
    d_rng = range(2)  # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = sm.tsa.arima.ARIMA(ts, order=(i, d, j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except:
                    continue

    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

    return best_aic, best_order, best_mdl


if __name__ == '__main__':
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2024, 1, 1)

    symbols = ['SPY', 'TLT', 'MSFT']

    # raw adjusted close prices
    data = pd.DataFrame({sym: get_px(sym) for sym in symbols})

    # log returns
    lrets = np.log(data / data.shift(1)).dropna()

    # Notice I’ve selected a specific time period to run this analysis
    TS = lrets.SPY.loc['2020':'2023']
    res_tup = _get_best_model(TS)
