import pandas as pd

class AlgoModel:
    def __init__(self):
        self.short_window = 50
        self.long_window = 200

    def train(self, historical_data):
        # Train logic if needed (e.g., parameter tuning)
        pass

    def predict(self, data):
        # Generate trading signals based on moving averages
        data['Short_MA'] = data['Close'].rolling(window=self.short_window).mean()
        data['Long_MA'] = data['Close'].rolling(window=self.long_window).mean()
        data['Signal'] = 0
        data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1
        data.loc[data['Short_MA'] <= data['Long_MA'], 'Signal'] = -1
        return data['Signal'].iloc[-1]
