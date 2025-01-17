import numpy as np
import pandas as pd
import requests
from models.algo_model import AlgoModel
from models.hft_model import HFTModel
from backtesting.backtest import Backtest
import time
import os
from datetime import datetime, timedelta

class TradingBot:
    def __init__(self, api_key, account_id, news_api_key):
        self.algo_model = AlgoModel()
        self.hft_model = HFTModel(api_key, account_id)
        self.trade_log = []
        self.news_api_key = news_api_key
        self.position_open = False
        self.highest_price = 0

    def train_models(self, historical_data, tick_data):
        self.algo_model.train(historical_data)

    def generate_signals(self, data):
        # Moving Average Crossover Strategy
        short_window = 40
        long_window = 100

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        # Create short simple moving average
        signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

        # Create long simple moving average
        signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

        # Create signals
        signals.loc[short_window:, 'signal'] = np.where(signals.loc[short_window:, 'short_mavg'] > signals.loc[short_window:, 'long_mavg'], 1.0, 0.0)

        # Generate trading orders
        signals['positions'] = signals['signal'].diff()

        return signals['positions'].iloc[-1]

    def execute_trades(self, signal, symbol):
        if signal == 1 and not self.position_open:
            self.hft_model.trade(signal, symbol)
            self.position_open = True
            self.highest_price = self.current_price
            self.log_trade(signal, symbol)
        elif signal == -1 and self.position_open:
            self.hft_model.trade(signal, symbol)
            self.position_open = False
            self.log_trade(signal, symbol)

    def log_trade(self, signal, symbol):
        trade = {
            'timestamp': pd.Timestamp.now(),
            'signal': signal,
            'symbol': symbol
        }
        self.trade_log.append(trade)
        print(f"Logged trade: {trade}")

    def generate_report(self):
        df = pd.DataFrame(self.trade_log)
        report_path = os.path.join(os.getcwd(), 'trading_bot_project', 'data', 'trade_report.csv')
        df.to_csv(report_path, index=False)
        print(f"Trade report saved to {report_path}")

    def fetch_market_news(self):
        # Using NewsAPI
        recent_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q=forex&from={recent_date}&sortBy=publishedAt&apiKey={self.news_api_key}"
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)
        print(f"Request URL: {url}")  # Debugging statement to check the request URL
        print(f"Response Status Code: {response.status_code}")  # Debugging statement to check the status code
        if response.status_code == 200:
            news_data = response.json()
            return news_data
        else:
            print(f"Failed to fetch news: {response.status_code}")
            print(f"Response Content: {response.content}")  # Debugging statement to check the response content
            return None

    def analyze_news(self, news_data):
        # Analyze news data to generate a sentiment score
        sentiment_score = 0
        if news_data and 'articles' in news_data:
            for article in news_data['articles']:
                if "positive" in article.get('description', ''):
                    sentiment_score += 1
                elif "negative" in article.get('description', ''):
                    sentiment_score -= 1
        else:
            print("No articles found in news data.")
        return sentiment_score

    def run(self):
        symbol = 'EUR_USD'
        take_profit_threshold = 0.001  # Example threshold for taking profit
        while True:
            try:
                # Fetch real-time price data
                self.current_price = self.hft_model.get_realtime_data(symbol)
                if self.current_price is None:
                    print("No price data available. Skipping this iteration.")
                    time.sleep(60)
                    continue
                print(f"Fetched real-time price for {symbol}: {self.current_price}")
                
                # Prepare the data for signal generation
                data = pd.DataFrame([[self.current_price]], columns=['Close'])
                
                # Generate trading signal
                signal = self.generate_signals(data)
                print(f"Generated trading signal: {signal}")
                
                # Fetch market news
                news_data = self.fetch_market_news()
                if news_data:
                    sentiment_score = self.analyze_news(news_data)
                    print(f"Market sentiment score: {sentiment_score}")
                    
                    # Adjust signal based on sentiment score
                    if sentiment_score > 0:
                        signal = 1  # Buy
                    elif sentiment_score < 0:
                        signal = -1  # Sell

                # Check for take-profit condition
                if self.position_open and self.current_price < self.highest_price - take_profit_threshold:
                    signal = -1  # Sell to take profit

                # Execute trade based on the generated signal
                self.execute_trades(signal, symbol)
                print(f"Executed trade based on signal: {signal}")

                # Update highest price if position is open
                if self.position_open:
                    self.highest_price = max(self.highest_price, self.current_price)
                
                # Wait before the next fetch
                time.sleep(60)  # Fetch data every minute
            except Exception as e:
                print(f"Error occurred: {e}")
                time.sleep(60)

if __name__ == "__main__":
    API_KEY = 'c6359bd1fedb480b229349ace651b8c8-19cfffc8771179b5e640bacd9ba7b498'
    ACCOUNT_ID = '101-001-30129969-001'
    NEWS_API_KEY = '1d48bf7f42864457a9214e188d1e8aa1'  # Replace with your NewsAPI key

    bot = TradingBot(API_KEY, ACCOUNT_ID, NEWS_API_KEY)
    historical_data_path = os.path.join(os.getcwd(), 'trading_bot_project', 'data', 'historical_data.csv')
    tick_data_path = os.path.join(os.getcwd(), 'trading_bot_project', 'data', 'tick_data.csv')
    
    # Check if the tick data file exists, if not print a message and proceed
    if not os.path.exists(tick_data_path):
        print(f"Warning: tick_data.csv not found at {tick_data_path}. Fetching data from the OANDA API.")
        tick_data = pd.DataFrame(columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])  # Empty placeholder
    else:
        tick_data = pd.read_csv(tick_data_path)
    
    # Load historical data
    if os.path.exists(historical_data_path):
        historical_data = pd.read_csv(historical_data_path)
    else:
        print(f"Warning: historical_data.csv not found at {historical_data_path}. Fetching data from OANDA.")
        historical_data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])  # Empty placeholder
    
    # Train the models
    bot.train_models(historical_data, tick_data)
    
    # Run the trading bot
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Stopping the trading bot and generating report...")
        bot.generate_report()