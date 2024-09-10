import numpy as np
from scipy.special import expit
from AutoRegression import *
from data_processing import min_max_normalize


def get_sample_from_dist(distribution, *args, **kwargs):
    return distribution(*args, **kwargs)


def generate_datapoint(data, config, model):
    error_delta = get_sample_from_dist(config.distribution, **config.dist_params)
    new_val = model.arima.step(data[-model.arima.p:], error_delta)
    return max(0, new_val)


def data_step(data, config, model):
    new_val = generate_datapoint(data, config, model)
    data = np.roll(data, -1)
    data[-1] = new_val

    # Update all indicators with the new data point
    for indicator in model.indicators:
        indicator.step(data)

    return data


def initialize_data(config, model):
    '''
    Initializes the dataset with values drawn from the specified distribution.

    :param config: Config object with initial settings for data generation.
    :param model: Model containing the ARIMA and indicators setup.
    :return: Numpy array of initial data points.
    '''

    data = [config.initial_value] * config.data_size
    for _ in range(config.data_size-1):
        new_val = generate_datapoint(data, config, model)
        data.pop(0)  # Remove oldest data point
        data.append(new_val)  # Add new data point

        # Update all indicators with the new data point
        for indicator in model.indicators:
            indicator.step(data)

    return np.array(data)


def run_simulation(iterations, data, model, config):
    print(f"Running {iterations} iterations...")
    dataset = []

    for _ in range(iterations):
        ''' Generate next data point, x[n+1] = x[n] + delta_x, where delta_x ~ N(0, 1) '''
        new_x = generate_datapoint(data, config, model)
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
        min_val, max_val = np.min(data), np.max(data)
        X = model.x_norm(min_val, max_val, mu)
        y = model.scaler.norm(delta_x)

        dataset.append((X, y))
        model.add_datapoint(X, y)

        ''' Advance signal by delta_x '''
        data = np.roll(data, -1)
        data[-1] = new_x

    model.end_run()

    return data, model, dataset
