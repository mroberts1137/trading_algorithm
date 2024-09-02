from Dist import Dist


class Model:
    def __init__(self, n_bins, arima):
        self.n_bins = n_bins
        # ARIMA - AutoRegression Integrated Moving Average Model
        self.arima = arima

        self.dist_grid = self.create_grid(n_bins)
        self.indicators = []

    # Function to recursively create the grid
    def create_grid(self, n_bins, current_indices=[]):
        if len(current_indices) == len(n_bins):
            return Dist(current_indices)
        return [self.create_grid(n_bins, current_indices + [i]) for i in range(n_bins[len(current_indices)])]

    # Function to access elements in the nested grid
    def grid_at(self, bin_coords):
        element = self.dist_grid
        for coord in bin_coords:
            element = element[coord]
        return element

    '''
    TODO:
        Create array for Dist central-moments
        Plot/average in (u, v) coordinates
    '''
