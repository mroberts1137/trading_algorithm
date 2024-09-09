import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt


def autocorrelation(x):
    """
    Calculate the autocorrelation of a 1D array.

    Args:
        x (array-like): Input array.

    Returns:
        array: Autocorrelation of the input array.
    """

    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]  # Return only the positive lags

data = pd.read_csv('data/SPX_1min_sample.csv')[['Close']].values

acf = sm.tsa.stattools.acf(data, nlags=2880)

# print(f'Numpy Autocorrelation: {autocorrelation(data)}')
# print(f'Statsmodels Autocorrelation: {acf}')

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(data, label='SPX', color='blue')
# plt.plot(predictions, label='Predicted Values', color='red', linestyle='--')

plt.title('SPX_1min_sample')
plt.xlabel('Time (minutes)')
plt.ylabel('Price')
plt.legend()
plt.show()

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(acf, label='ACF', color='blue')
# plt.plot(predictions, label='Predicted Values', color='red', linestyle='--')

plt.title('AutoCorrelation Function of SPX_1min_sample')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.legend()
plt.show()
