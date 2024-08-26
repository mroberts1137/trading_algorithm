from Dist import Dist


class Model:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.dist_grid = self.create_grid(n_bins)

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
