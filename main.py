import numpy as np
from scipy.special import expit, logit
from data_generation import initialize_data, run_simulation, data_step
from data_visualization import *
from Config import Config
from Model import Model
from AutoRegression import AutoRegression
from Indicator import *
from Scaler import Scaler
from data_processing import min_max_normalize
from fit_arima import fit_arima


def test(data, model, trials):
    new_data = []
    predictions = []
    errors = []
    pred_deltas = []
    deltas = []
    loss = 0

    for _ in range(trials):
        pred_delta, pred_x, _ = model.predict_next(data)

        prev_x = data[-1]
        data = data_step(data, config, model)
        new_x = data[-1]

        error = pred_x - new_x
        delta_x = new_x - prev_x

        deltas.append(delta_x)
        pred_deltas.append(pred_delta)

        new_data.append(new_x)
        predictions.append(pred_x)

        errors.append(error)

        loss += error ** 2

        # print(f'Data: {new_x:.2f}, Prediction: {pred_x:.2f}')
        # print(f'Error: {error:.2f}, {(error / new_x * 100):.1f}%')

    return loss, errors, new_data, predictions


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
    # print(arima)
    # arima.plot_roots()

    lambdas = [50, 5]  # Window sizes for SMAs
    indicators = [SMA(window=l) for l in lambdas]
    n_bins = [10, 10]  # Number of bins for each indicator
    y_bins = 10  # Number of bins for y values

    x_norm = min_max_normalize
    scaler = Scaler(expit, logit)

    model = Model(n_bins, y_bins, arima, indicators, x_norm, scaler)

    data = initialize_data(config, model)

    '''
    Train Model
    '''
    iterations = 1000
    data, model, dataset = run_simulation(iterations, data, model, config)

    # print(model.get_dist([5, 5]))
    # print(model.get_dist([4, 6]))
    # print(model.get_dist([6, 4]))

    # visualize_results(data, config, model, dataset)

    '''
    Test Model
    '''

    trials = 100
    loss, errors, new_data, predictions = test(data, model, trials)
    print(f'Loss: {loss}')
    print(f'Residuals: Mean: {np.mean(errors)}, Stdv: {np.std(errors)}')

    plot_results(new_data, predictions, errors)
    plot_residues(errors)

    # Second round of training

    iterations = 10000
    data, model, _ = run_simulation(iterations, data, model, config)

    trials = 100
    loss, errors, new_data, predictions = test(data, model, trials)
    print(f'Loss: {loss}')
    print(f'Residuals: Mean: {np.mean(errors)}, Stdv: {np.std(errors)}')

    plot_results(new_data, predictions, errors)
    plot_residues(errors)


    '''
    TODO:
    [ ] Batch data - create random data, length n
    [ ] Feed batch data into model
    [ ] Use same data to train neural network model
    [ ] Compare models
    '''
