class Config:
    def __init__(self, distribution, dist_params, data_size, initial_value, lambdas, n_bins):
        self.distribution = distribution
        self.dist_params = dist_params
        self.data_size = data_size
        self.initial_value = initial_value
        self.lambdas = lambdas
        self.n_bins = n_bins
