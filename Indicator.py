import numpy as np


class Indicator:
    def __init__(self):
        self.current_val = None
        self.name = ''
        self.data = []

    def get_value(self):
        pass

    def step(self):
        '''
        Update indicator with next data step
        :param data: updated data
        :return: updated indicator value
        '''
        self.data.append(self.current_val)


class SMA(Indicator):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.current_val = 0
        self.name = f'SMA({self.window})'
        # It might be a good idea to have a range or bin count here

    def get_value(self, data):
        if len(data) >= self.window:
            return np.sum(data[-self.window:]) / float(self.window)

    def step(self, data):
        self.current_val = self.get_value(data)
        super().step()
        return self.current_val


class Beta(Indicator):
    def __init__(self, polynomial_order, order):
        super().__init__()
        self.polynomial_order = polynomial_order
        self.order = order
        self.beta = np.zeros(self.polynomial_order + 1)
        self.name = f'Beta_{self.order}'

    def get_value(self, data):
        pass

    def update_beta(self, lambdas, mu):
        self.beta = np.polynomial.polynomial.Polynomial.fit(lambdas, mu, self.polynomial_order)

    def step(self, data):
        lambdas = None
        mu = None
        super().step()
        return self.beta[self.order]
