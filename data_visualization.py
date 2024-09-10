import matplotlib.pyplot as plt
import numpy as np


def plot_data_with_indicators(data, indicators, data_size):
    fig, ax = plt.subplots()
    ax.plot(data)
    for indicator in indicators:
        ax.plot(indicator.data[-data_size:])
    plt.show()


def plot_indicator_space_and_heatmap(model, dataset):
    X, y = zip(*dataset) if dataset else ([], [])
    X, y = np.array(X), np.array(y)

    # Setting up the figure for two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Scatter plot of the indicator space
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='none', s=10)
    ax1.set_title('Indicator Space')
    ax1.set_xlabel(model.indicators[0].name)
    ax1.set_ylabel(model.indicators[1].name)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    plt.colorbar(scatter, ax=ax1, label='Normalized Change (y)')

    # Adding grid lines for visual aid but not for binning
    for i in range(model.n_bins[0] + 1):
        x = i / model.n_bins[0]
        ax1.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for j in range(model.n_bins[1] + 1):
        y = j / model.n_bins[1]
        ax1.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    # Preparing data for heatmap from Dist objects
    mean_y_values = np.zeros((model.n_bins[0], model.n_bins[1]))
    for coord in model.bin_coords:
        dist = model.get_dist(coord)
        if dist and dist.count.sum() > 0:  # Only if the distribution has data
            mean_y_values[coord] = dist.mean

    # Heatmap of mean y values in each bin
    im = ax2.imshow(mean_y_values.T, origin='lower', cmap='viridis', extent=[0, 1, 0, 1], vmin=0, vmax=1)
    ax2.set_title('Heatmap of Mean y Values')
    ax2.set_xlabel(model.indicators[0].name)
    ax2.set_ylabel(model.indicators[1].name)
    plt.colorbar(im, ax=ax2, label='Mean of y')

    # Setting up the grid for clarity in heatmap
    ax2.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xticks(np.linspace(0, 1, model.n_bins[0] + 1))
    ax2.set_yticks(np.linspace(0, 1, model.n_bins[1] + 1))

    plt.tight_layout()
    plt.show()


def visualize_results(data, config, model, dataset):
    plot_data_with_indicators(data, model.indicators, config.data_size)
    plot_indicator_space_and_heatmap(model, dataset)


#
#
#
#
#
# def plot_data_and_indicators(data, indicators, data_size):
#     fig, ax = plt.subplots()
#     ax.plot(data)
#     for indicator in indicators:
#         ax.plot(indicator.data[-data_size:])
#     plt.show()
#
#
# def visualize_binning_and_distribution(X, y, n_bins, indicator_names):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#
#     # Scatter Plot
#     scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', vmin=0, vmax=1)
#     ax1.set_xlabel(indicator_names[0])
#     ax1.set_ylabel(indicator_names[1])
#     ax1.set_title('Indicator Space')
#     plt.colorbar(scatter, ax=ax1)
#
#     # Adding grid lines
#     for ax in [ax1, ax2]:
#         x_bins = np.linspace(0, 1, n_bins[0] + 1)
#         y_bins = np.linspace(0, 1, n_bins[1] + 1)
#         for x in x_bins:
#             ax.vlines(x, ymin=0, ymax=1, colors='gray', linestyles='dashed', linewidth=0.5)
#         for y in y_bins:
#             ax.hlines(y, xmin=0, xmax=1, colors='gray', linestyles='dashed', linewidth=0.5)
#
#     # Heatmap
#     bin_y_values = {}
#     for i in range(len(X)):
#         bin_idx = (np.digitize(X[i, 0], x_bins) - 1, np.digitize(X[i, 1], y_bins) - 1)
#         if bin_idx not in bin_y_values:
#             bin_y_values[bin_idx] = []
#         bin_y_values[bin_idx].append(y[i])
#
#     mean_y_values = np.array(
#         [[np.mean(bin_y_values.get((i, j), [np.nan])) for j in range(n_bins[1])] for i in range(n_bins[0])])
#     heatmap = ax2.imshow(mean_y_values.T, origin='lower', cmap='viridis', extent=[0, 1, 0, 1], aspect='auto')
#     ax2.set_title('Heatmap of Mean y Values in Each Bin')
#     ax2.set_xlabel(indicator_names[0])
#     ax2.set_ylabel(indicator_names[1])
#     plt.colorbar(heatmap, ax=ax2)
#
#     plt.show()

# Minor optimization: Combining the two plotting functions into one might be useful for consistency in plots