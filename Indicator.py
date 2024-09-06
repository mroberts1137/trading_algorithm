import numpy as np


class Indicator:
    def __init__(self):
        self.current_val = None
        self.name = ''

    def step(self, data):
        '''
        Update indicator with next data step
        :param data: updated data
        :return: updated indicator value
        '''
        pass


class SMA(Indicator):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.current_val = 0
        self.name = f'SMA({self.window})'
        # It might be a good idea to have a range or bin count here

    def step(self, data):
        if len(data) >= self.window:
            self.current_val = np.sum(data[-self.window:]) / float(self.window)
            return self.current_val


class Beta(Indicator):
    def __init__(self, polynomial_order, order):
        super().__init__()
        self.polynomial_order = polynomial_order
        self.order = order
        self.beta = np.zeros(self.polynomial_order + 1)
        self.name = f'Beta_{self.order}'

    def update_beta(self, lambdas, mu):
        self.beta = np.polynomial.polynomial.Polynomial.fit(lambdas, mu, self.polynomial_order)

    def step(self, lambdas, mu):
        return self.beta[self.order]
