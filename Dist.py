import numpy as np
import matplotlib.pyplot as plt


class Dist:
    def __init__(self, domain):
        self.domain = domain
        self.min = 0
        self.max = 1
        self.bins = 10
        self.count = np.zeros(self.bins, dtype=int)

    def __repr__(self):
        # Create a histogram plot
        plt.bar(range(self.bins), self.count, width=1, edgecolor='black')
        plt.title(f"D({self.domain})")
        plt.xlabel('y')
        plt.ylabel('Count')
        plt.show()
        return f"Dist({self.domain})"

    def add(self, val):
        # Clamp val between [min, max]
        val = max(self.min, min(self.max, val))
        bin = int(np.floor(val * self.bins))
        self.count[bin] += 1
