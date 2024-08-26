class Indicator:
    def __init__(self):
        pass

    def step(self, data):
        '''
        Update indicator with next data step
        :param data: updated data
        :return: updated indicator value
        '''
        return

class SMA(Indicator):
    def __init__(self, window):
        super().__init__(self)
        self.window = window