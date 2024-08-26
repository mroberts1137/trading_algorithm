# from typing import List, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from Config import Config
from Dist import Dist
from Indicator import *
from Model import Model


def get_sample_from_dist(distribution, *args, **kwargs):
    '''
    Get the next data point drawn from a distribution
    :param data: Dataset for which to generate next data point
    :param dist: The distribution to sample for the data point
    :return: The sampled data point
    '''
    return distribution(*args, **kwargs)


def transform_moving_average(data, window):
    '''
    Transform data set into the indicators
    :param data: The dataset to transform
    :param indicators: List of indicators (e.g. SMA, EMA)
    :return: List of indicator values for each indicator
    '''
    return np.ma.average(data)


def initialize_data(size, x0, distribution, *args, **kwargs):
    '''
    Create an initial dataset by drawing points from dist
    :param size: Number of data points to generate
    :param x0: Starting value for data
    :param distribution: Distribution to sample
    :return: Array of data points
    '''
    x = [x0]
    for i in range(size-1):
        new_val = max(0, x[i] + distribution(*args, **kwargs))
        x.append(new_val)

    return np.array(x)


def simple_moving_average(data, window):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def current_sma(x, window):
    return np.sum(x[-window:]) / float(window)


def correcting_normal_dist(x, x0, *args, **kwargs):
    if x > x0:
        return np.random.normal(-1, 1)
    else:
        return np.random.normal(1, 1)


def min_max_normalize(min_val, max_val, value):
    return (value - min_val) / (max_val - min_val)


# Function to recursively create the grid
def create_grid(n_bins, current_indices=[]):
    if len(current_indices) == len(n_bins):
        return Dist(current_indices)
    return [create_grid(n_bins, current_indices + [i]) for i in range(n_bins[len(current_indices)])]


# Function to access elements in the nested grid
def access_grid_element(grid, bin_coords):
    element = grid
    for coord in bin_coords:
        element = element[coord]
    return element


def run_simulation(iterations, data, model, config):
    print(f"Running {iterations} iterations...")

    for _ in range(iterations):
        ''' Generate next data point, x[n+1] = x[n] + delta_x, where delta_x ~ N(0, 1) '''
        delta_x = get_sample_from_dist(config.distribution, **config.dist_params)

        ''' Compute indicators, e.g. mu(x) = SMA(x) '''
        # for indicator in indicators:
        #   indicator.step(data)
        mu = [current_sma(data, window) for window in config.lambdas]

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
        x_new = data[-1] + delta_x
        data = np.roll(data, -1)
        data[-1] = x_new

    print("Simulation complete.")


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

    lambdas = [50, 10]

    data = initialize_data(data_size, initial_value, distribution, **dist_params)
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show()

    dataset = []
    n_bins = [10, 10]
    bins = np.zeros(shape=n_bins)

    # Create the grid of Distributions
    model = Model(n_bins)
    # dist_grid = create_grid(n_bins)

    config = Config(distribution=distribution, dist_params=dist_params, data_size=data_size, initial_value=initial_value, lambdas=lambdas, n_bins=n_bins)

    '''
    BEGIN DATA GENERATION
    '''
    run_simulation(10000, data, model, config)
    '''
    END DATA GENERATION
    '''

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
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
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
    plt.imshow(mean_y_values.T, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
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
