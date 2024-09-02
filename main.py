# from typing import List, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from Config import Config
from Dist import Dist
from Indicator import *
from Model import Model
from AutoRegression import *


def get_sample_from_dist(distribution, *args, **kwargs):
    '''
    Get the next data point drawn from a distribution
    :param data: Dataset for which to generate next data point
    :param dist: The distribution to sample for the data point
    :return: The sampled data point
    '''
    return distribution(*args, **kwargs)


def data_step(data, config, model):
    error_delta = get_sample_from_dist(config.distribution, **config.dist_params)
    new_val = model.arima.step(data[-model.arima.p:], error_delta)
    new_val = max(0, new_val)

    return new_val


def initialize_data(config, model):
    '''
    Create an initial dataset by drawing points from dist
    :param size: Number of data points to generate
    :param x0: Starting value for data
    :param autoregression: AutoRegression Model for generating new data points
    :param distribution: Distribution to sample
    :return: Array of data points
    '''
    data = [config.initial_value] * config.data_size
    for _ in range(config.data_size-1):
        new_val = data_step(data, config, model)
        data.pop(0)
        data.append(new_val)

    return np.array(data)


def min_max_normalize(min_val, max_val, value):
    return (value - min_val) / (max_val - min_val)


def run_simulation(iterations, data, model, config):
    print(f"Running {iterations} iterations...")

    for _ in range(iterations):
        ''' Generate next data point, x[n+1] = x[n] + delta_x, where delta_x ~ N(0, 1) '''
        new_x = data_step(data, config, model)
        delta_x = new_x - data[-1]

        ''' Compute indicators, e.g. mu(x) = SMA(x) '''
        for indicator in model.indicators:
            indicator.step(data)
        mu = [indicator.current_val for indicator in model.indicators]

        ''' 
        Normalize (indicators, delta_x) to get (X, y) 
            X normalization: min-max normalization: (x_min, x_max) -> (0, 1)
            NOTE: x_min/max only works for indicators in same domain as x(t). Use indicator_min/max in general.
            y-normalization: logistic sigmoid: (-infty, infty) -> (0, 1)
        '''
        min_val = np.min(data)
        max_val = np.max(data)
        X = min_max_normalize(min_val, max_val, mu)
        y = expit(delta_x)
        dataset.append((X, y))

        ''' 
        Bin X in feature space. i.e. X -> U = Bin(X) 
            NOTE: floor(X[i] * n_bins[i]) relies on X[i] in [0, 1)
        '''
        bin_coords = tuple([int(np.floor(X[i] * n_bins[i])) for i in range(len(n_bins))])

        ''' Bin y in distribution D(U) where U = Bin(X) '''
        dist = model.grid_at(bin_coords)
        dist.add(y)

        ''' Advance signal by delta_x '''
        data = np.roll(data, -1)
        data[-1] = new_x

    print("Simulation complete.")

    return data, model


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
    distribution = np.random.normal
    dist_params = {"loc": 0, "scale": 1}
    data_size = 100
    initial_value = 100

    config = Config(distribution=distribution, dist_params=dist_params, data_size=data_size,
                    initial_value=initial_value)

    ''' AutoRegression Model '''
    ar_p = 0
    ar_q = 10
    ar_phi = np.full(ar_p, 1)
    ar_theta = np.full(ar_q, 1)
    arima = AutoRegression(ar_phi, ar_theta)
    print(arima)
    arima.plot_roots()

    dataset = []
    lambdas = [50, 10]
    n_bins = [10, 10]

    # Create the model
    model = Model(n_bins, arima)

    for l in lambdas:
        model.indicators.append(SMA(window=l))

    data = initialize_data(config, model)

    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show()

    '''
    BEGIN DATA GENERATION
    '''
    data, model = run_simulation(10000, data, model, config)
    '''
    END DATA GENERATION
    '''

    print(model.dist_grid[4][6])
    print(model.dist_grid[6][4])

    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            print(f'Dist[{i}, {j}]: {model.dist_grid[i][j].count}')

    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show()

    '''
    Post-simulation analysis
    '''

    # Extract X and y values
    X = np.array([item[0] for item in dataset])
    y = np.array([item[1] for item in dataset])

    ''' Bin y-data '''

    # Calculate bin edges
    x_bins = np.linspace(0, 1, n_bins[0] + 1)
    y_bins = np.linspace(0, 1, n_bins[1] + 1)

    # Find the bin index for each data point
    # U
    x_bin_indices = np.digitize(X[:, 0], bins=x_bins) - 1
    y_bin_indices = np.digitize(X[:, 1], bins=y_bins) - 1

    # Aggregate y values for each bin
    # D(U)
    bin_y_values = {}

    for i in range(len(X)):
        bin_idx = (x_bin_indices[i], y_bin_indices[i])
        if bin_idx not in bin_y_values:
            bin_y_values[bin_idx] = []
        bin_y_values[bin_idx].append(y[i])
    # print(f'bin_y_values: {bin_y_values}')

    # Initialize a 2D array to store the mean y values for each bin
    mean_y_values = np.full((n_bins[0], n_bins[1]), np.nan)

    # Create probability distributions for each bin
    # D(U)
    bin_distributions = {}

    for bin_idx, y_values in bin_y_values.items():
        hist, bin_edges = np.histogram(y_values, bins='auto', density=True)
        bin_distributions[bin_idx] = (hist, bin_edges)

    # Calculate the mean y value for each bin
    for bin_idx, y_values in bin_y_values.items():
        mean_y_values[bin_idx] = np.mean(y_values)

    # Plot the data with the grid
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='y value')
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Indicator Space')

    # Draw vertical grid lines
    for x in x_bins:
        plt.vlines(x, ymin=0, ymax=1, colors='gray', linestyles='dashed', linewidth=0.5)

    # Draw horizontal grid lines
    for y in y_bins:
        plt.hlines(y, xmin=0, xmax=1, colors='gray', linestyles='dashed', linewidth=0.5)

    plt.show()

    # Plot the heatmap
    plt.imshow(mean_y_values.T, origin='lower', cmap='viridis', extent=[0, 1, 0, 1], vmin=0, vmax=1)
    plt.colorbar(label='Mean y value')
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.title('Heatmap of Mean y Values in Each Bin')
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)

    # Draw grid lines
    plt.xticks(np.linspace(0, 1, n_bins[0] + 1))
    plt.yticks(np.linspace(0, 1, n_bins[1] + 1))
    plt.show()

    # Print the probability distributions for each bin
    # for bin_idx, (hist, bin_edges) in bin_distributions.items():
    #     print(f'Bin {bin_idx}:')
    #     print(f'  Histogram: {hist}')
    #     print(f'  Bin edges: {bin_edges}')
