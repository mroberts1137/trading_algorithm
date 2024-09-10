import numpy as np
from data_generation import initialize_data, run_simulation
from data_visualization import visualize_results
from Config import Config
from Model import Model
from AutoRegression import AutoRegression
from Indicator import *
from fit_arima import fit_arima

if __name__ == "__main__":
    '''
    Initialize CONFIG options for simulation
    - distribution: default distribution to generate sample data
    - dist_params: distribution parameters, loc = mean, scale = stdv for Gaussian dist
    - data_size: number of data points to retain in signal data, x(t)
    - initial_value: initial data value for sample data
    - lambdas: array of SMA window periods. Bound between [1, data_size]
    - data: signal data, x(t)
    - dataset: input/output pairs (X, y) for each generated sample data point
    - n_bins: array of bin number for each indicator. Used for binning X -> U = Bin(X) -> D(U)
    - bins: array of bins for indicator space, U = [i, j, k, ...]
    - dist_grid: grid of distributions, D(U) where coords are bins U. D(U) = histogram of y-values given X in U
    '''

    # LOAD FILE:
    # rawdata_dir = os.path.join(os.path.join(os.getcwd(), 'load_dataset'), 'raw_data')
    # data_file = os.path.join(self.rawdata_dir, x_data)

    # Setup config, model, and initial data
    distribution = np.random.normal
    dist_params = {"loc": 0, "scale": 1}
    data_size = 100
    initial_value = 100

    config = Config(distribution=distribution, dist_params=dist_params, data_size=data_size,
                    initial_value=initial_value)

    ''' 
    AutoRegression Integrated Moving Average (ARIMA) Model:
    ARIMA(p, d, q)
    '''
    ar_p, ar_q = 50, 5
    ar_phi = np.full(ar_p, 1 / ar_p)
    ar_theta = np.full(ar_q, 1)
    arima = AutoRegression(ar_phi, ar_theta)
    print(arima)
    arima.plot_roots()

    lambdas = [50, 5]  # Window sizes for SMAs
    indicators = [SMA(window=l) for l in lambdas]
    n_bins = [10, 10]  # Number of bins for each indicator
    y_bins = 10  # Number of bins for y values

    model = Model(n_bins, y_bins, arima, indicators)

    data = initialize_data(config, model)

    # Run simulation
    iterations = 10000
    data, model, dataset = run_simulation(iterations, data, model, config)

    print(model.get_dist([5, 5]))
    print(model.get_dist([4, 6]))
    print(model.get_dist([6, 4]))

    visualize_results(data, config, model, dataset)
