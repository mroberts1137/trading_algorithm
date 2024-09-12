import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import beta


class Dist:
    def __init__(self, domain, y_bins):
        self.domain = domain    # Array of integer grid indices
        self.min = 0
        self.max = 1

        self.y_bins = y_bins
        self.y_range = np.linspace(self.min, self.max, self.y_bins)
        self.bin_width = (self.max - self.min) / self.y_bins

        self.count = np.zeros(self.y_bins, dtype=int)

        self.max_moment = 4
        self.moments = np.zeros(self.max_moment)
        # Initialize moments to be Gaussian
        self.moments[0] = (self.max - self.min) / 2
        self.moments[1] = (self.max - self.min) / 4

        self.mean = self.moments[0]
        self.stdv = np.sqrt(self.moments[1])
        self.skewness = self.moments[2] / self.stdv ** 3
        self.kurtosis = self.moments[3] / self.stdv ** 4

        self.quantiles = [0, 0, 0]

        self.dist_fit = norm(loc=self.moments[0], scale=np.sqrt(self.moments[1]))

        self.stats = {"mean": self.mean, "stdv": self.stdv, "skewness": self.skewness, "kurtosis": self.kurtosis}
        self.describe = [f"{key}: {self.stats[key]:.4f}" for key in self.stats]

    def __repr__(self):
        '''
        Create a histogram plot
        use `print(Dist(U))` to plot histogram
        :return: string representation (__repr__) of object to print
        '''
        self.plot_distribution()
        return f"Dist({self.domain})"

    def add(self, val):
        # Clamp val between [min, max]
        val = max(self.min, min(self.max, val))

        y_bin = int(np.floor(val * self.y_bins))
        self.count[y_bin] += 1

    def calculate_moments(self):
        if np.sum(self.count) > 0:
            mean = np.average(self.y_range, weights=self.count)
            self.moments[0] = mean

            for order in range(2, self.max_moment + 1):
                self.moments[order - 1] = np.average((self.y_range - mean) ** order, weights=self.count)

            # Ensure the variance is not zero (this happens if only 1 bin has data)
            if self.moments[1] == 0:
                self.moments[1] = (self.max - self.min) / 4

            self.mean = self.moments[0]
            self.stdv = np.sqrt(self.moments[1])
            self.skewness = self.moments[2] / self.stdv ** 3
            self.kurtosis = self.moments[3] / self.stdv ** 4

    def calculate_quantiles(self):
        # Cumulative sum of the counts in each bin
        cumulative_counts = np.cumsum(self.count)

        # Total number of data points
        total_count = cumulative_counts[-1]

        # Quartile thresholds
        q1_thresh = 0.25 * total_count
        q2_thresh = 0.50 * total_count
        q3_thresh = 0.75 * total_count

        q1 = q2 = q3 = None

        for i, sum in enumerate(cumulative_counts):

            # Track the first time we exceed each threshold
            if q1 is None and sum >= q1_thresh:
                q1 = self.y_range[i]

            if q2 is None and sum >= q2_thresh:
                q2 = self.y_range[i]

            if q3 is None and sum >= q3_thresh:
                q3 = self.y_range[i]
                break  # We can stop once we find Q3

        quantiles = [q1, q2, q3]
        min_val = self.min + self.bin_width / 2
        max_val = self.max - self.bin_width / 2
        # Take midpoint of y_bins and clamp between (min_val, max_val)
        self.quantiles = [np.maximum(min_val, np.minimum(max_val, q + self.bin_width / 2)) for q in quantiles]

    def end_run(self, print_stats=False):
        self.calculate_moments()
        self.calculate_quantiles()
        self.reconstruct_distribution()

        self.stats = {"mean": self.mean, "stdv": self.stdv, "skewness": self.skewness, "kurtosis": self.kurtosis}
        self.describe = [f"{key}: {self.stats[key]:.4f}" for key in self.stats]

        if print_stats:
            print(f'Dist{self.domain}: {self.count}\t{self.describe}')

    def reconstruct_distribution(self):
        a = self.mean * ((self.mean * (self.mean + 1)) / self.stdv ** 2 + 1)
        b = (self.mean * (self.mean + 1)) / self.stdv ** 2 + 2
        self.dist_fit = norm(loc=self.mean, scale=self.stdv)
        # self.dist_fit = self.beta_prime_pdf(x, a, b)

    def beta_prime_pdf(self, x, a, b):
        return (x**(a - 1) * (1 + x)**(-a - b)) / beta(a, b)

    def sample(self):
        min_val = self.min + self.bin_width / 2
        max_val = self.max - self.bin_width / 2
        sample_val = self.dist_fit.rvs()
        val = np.maximum(min_val, np.minimum(max_val, sample_val))
        return val

    def plot_distribution(self):
        # Generate y values and the corresponding PDF values
        y_values = np.linspace(self.min, self.max, 1000)
        pdf_values = self.dist_fit.pdf(y_values)

        plt.figure(figsize=(8, 5))

        # Plot Data
        if np.sum(self.count) > 0:
            bin_width = (self.max - self.min) / self.y_bins
            normalized_count = (self.count / np.sum(self.count)) / bin_width
            plt.bar(self.y_range, normalized_count, width=bin_width, edgecolor='black')

        # Plot Fit
        plt.plot(y_values, pdf_values, label=f'N({self.mean:.4f}, {self.stdv:.4f}^2)', color='red')

        plt.title(f'D{self.domain}: {self.describe}')
        plt.xlabel('y')
        plt.ylabel('p(y)')
        plt.legend()
        plt.show()
