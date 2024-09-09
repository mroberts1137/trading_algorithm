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

        self.count = np.zeros(self.y_bins, dtype=int)

        self.max_moment = 4
        self.moments = np.zeros(self.max_moment)
        # Initialize moments to be Gaussian, N(.5, 1) -> mu_2 = sigma^2, mu_4 = 3 sigma^4
        self.moments[0] = 0.5
        self.moments[1] = 1
        self.moments[3] = 3

        self.mean = self.moments[0]
        self.stdv = np.sqrt(self.moments[1])
        self.skewness = self.moments[2] / self.stdv ** 3
        self.kurtosis = self.moments[3] / self.stdv ** 4

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
                self.moments[1] = 1

            self.mean = self.moments[0]
            self.stdv = np.sqrt(self.moments[1])
            self.skewness = self.moments[2] / self.stdv ** 3
            self.kurtosis = self.moments[3] / self.stdv ** 4

    def end_run(self):
        self.calculate_moments()
        self.reconstruct_distribution()

        self.stats = {"mean": self.mean, "stdv": self.stdv, "skewness": self.skewness, "kurtosis": self.kurtosis}
        self.describe = [f"{key}: {self.stats[key]:.4f}" for key in self.stats]

        print(f'Dist{self.domain}: {self.count}\t{self.describe}')

    def reconstruct_distribution(self):
        a = self.mean * ((self.mean * (self.mean + 1)) / self.stdv ** 2 + 1)
        b = (self.mean * (self.mean + 1)) / self.stdv ** 2 + 2
        self.dist_fit = norm(loc=self.mean, scale=self.stdv)
        # self.dist_fit = self.beta_prime_pdf(x, a, b)

    def beta_prime_pdf(self, x, a, b):
        return (x**(a - 1) * (1 + x)**(-a - b)) / beta(a, b)

    def sample(self):
        return self.dist_fit.rvs()

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
