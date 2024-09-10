import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, indicators, range=100):
    fig, ax = plt.subplots()
    ax.plot(data[-range:])
    for indicator in indicators:
        ax.plot(indicator.data[-range:])
    plt.show()


def plot_with_prediction(data, indicators, range=100, pred=None, quartiles=None):
    fig, ax = plt.subplots()

    # Plot the main data and indicators
    ax.plot(data[-range:], label='Data', color='blue')
    for i, indicator in enumerate(indicators):
        ax.plot(indicator.data[-range:], label=indicator.name, linestyle='--')

    # The x position where the prediction and box plot will appear (just after the last data point)
    x_pos = len(data[-range:]) + 1

    # Overlay the predicted value as a red dot at the front of the time series
    if pred is not None:
        ax.plot(x_pos, pred, 'ro', label='Prediction')  # Prediction dot

    # Draw a box plot at the front of the data using the quartiles
    if quartiles is not None:
        q1, q2, q3 = quartiles
        box_width = 3  # Width of the box plot

        # Draw the vertical lines for the whiskers and the box
        ax.vlines(x_pos - box_width / 2, q1, q3, color='lightblue', linewidth=2, label='Quartile Range')
        ax.vlines(x_pos + box_width / 2, q1, q3, color='lightblue', linewidth=2)

        # Draw the horizontal box edges at the quartiles
        ax.hlines(quartiles, x_pos - box_width / 2, x_pos + box_width / 2, color='lightblue', linewidth=2)

    # Set x-axis limits so the prediction and quartile box plot show on the right
    ax.set_xlim(0, x_pos + 5)  # Extend the x-axis a bit to give space for the plot

    # Add legend and show the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move the legend outside the plot
    plt.tight_layout()  # Adjust layout so everything fits well
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
    plot_data(data, model.indicators, config.data_size)
    plot_indicator_space_and_heatmap(model, dataset)
