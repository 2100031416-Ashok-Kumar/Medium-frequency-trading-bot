import numpy as np

class Backtest:
    def __init__(self, algo_model, hft_model):
        self.algo_model = algo_model
        self.hft_model = hft_model

    def run(self, historical_data, tick_data):
        # Backtest the combined strategies
        pass