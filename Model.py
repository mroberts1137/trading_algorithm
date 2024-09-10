import itertools
import numpy as np
from Dist import Dist


class Model:
    def __init__(self, n_bins, y_bins, arima, indicators, x_norm, scaler):
        self.indicators = indicators
        self.n_bins = n_bins
        self.y_bins = y_bins
        self.x_norm = x_norm
        self.scaler = scaler

        # ARIMA - AutoRegression Integrated Moving Average Model
        self.arima = arima

        self.dist_grid = {}

        # Initialize all possible bin coordinate combinations
        self.bin_coords = list(itertools.product(*[range(n) for n in self.n_bins]))
        for bin_coord in self.bin_coords:
            self.dist_grid[bin_coord] = Dist(bin_coord, self.y_bins)  # Assuming Dist is a class

    def get_dist(self, bin_coord):
        return self.dist_grid.get(tuple(bin_coord), None)

    def add_datapoint(self, x, y):
        bin_coord = tuple([int(np.floor(x[i] * self.n_bins[i])) for i in range(len(self.n_bins))])
        dist = self.get_dist(bin_coord)
        dist.add(y)

    def end_run(self):
        for bin_coord in self.bin_coords:
            dist = self.get_dist(bin_coord)
            dist.end_run()

        print("Simulation complete.")

    def predict_next(self, data):
        '''
        Predict next data point x[t+1] given previous data {x[t-T], ..., x[t]}
        :param data: input data array
        :return: predicted next data point
        '''
        mu = [indicator.current_val for indicator in self.indicators]

        min_val, max_val = np.min(data), np.max(data)
        X = self.x_norm(min_val, max_val, mu)

        bin_coord = tuple([int(np.floor(X[i] * self.n_bins[i])) for i in range(len(self.n_bins))])
        dist = self.get_dist(bin_coord)

        y_pred = dist.sample()
        y_quartiles = dist.quartiles

        pred = self.scaler.inverse(y_pred) + data[-1]
        quartiles = self.scaler.inverse(y_quartiles) + data[-1]

        return pred, quartiles


    '''
    TODO:
        [x] Create array for Dist central-moments
        [ ] Plot/average in (u, v) coordinates
        [ ] Add Box Plot in front of Price Data for Dist(X)
        [ ] Create LIVE PRICE PLOT -> update live when given input data
        [ ] Monte Carlo forecast
        [ ] Create buy/sell/hold ActionSignal: x_forecast -> trade_signal
        [ ] Input real data
        [ ] Beta Indicator
        [ ] Volatility Indicator
    '''
    '''
    TODO:
    - Residual Plots - a good model fit should have white-noise residuals - no correlations
    '''
