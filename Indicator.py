import numpy as np


class Indicator:
    def __init__(self):
        self.current_val = None

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

    def step(self, data):
        if len(data) >= self.window:
            self.current_val = np.sum(data[-self.window:]) / float(self.window)
            return self.current_val
