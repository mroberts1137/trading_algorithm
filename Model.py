import itertools
import numpy as np
from Dist import Dist


class Model:
    def __init__(self, n_bins, y_bins, arima, indicators):
        self.indicators = indicators
        self.n_bins = n_bins
        self.y_bins = y_bins

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
