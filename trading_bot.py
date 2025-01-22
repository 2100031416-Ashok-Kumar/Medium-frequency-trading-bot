import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.hft_model import HFTModel
import time
import os
from datetime import datetime, timedelta

class TradingBot:
    def __init__(self, api_key, account_id, news_api_key):
        self.hft_model = HFTModel(api_key, account_id)
        self.trade_log = []
        self.news_api_key = news_api_key
        self.position_open = False
        self.highest_price = 0
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.historical_data = pd.DataFrame()

    def train_model(self, data):
        # Prepare features and labels
        features = data.select_dtypes(include=[np.number]).drop(columns=['Close'])
        labels = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy}")

    def generate_features(self, data):
        # Technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])

        # Sentiment analysis
        sentiment_score = self.fetch_and_analyze_news()
        data['Sentiment'] = sentiment_score

        # Fill missing values
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        return data

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, series, short_period=12, long_period=26, signal_period=9):
        short_ema = series.ewm(span=short_period, adjust=False).mean()
        long_ema = series.ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, macd_signal

    def fetch_and_analyze_news(self):
        news_data = self.fetch_market_news()
        sentiment_score = self.analyze_news(news_data)
        return sentiment_score

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

        # Log trade to CSV file
        trade_df = pd.DataFrame([trade])
        trade_df.to_csv('trade_log.csv', mode='a', header=not os.path.exists('trade_log.csv'), index=False)

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

    def fetch_additional_historical_data(self, symbol, granularity, count):
        url = f"https://api-fxpractice.oanda.com/v3/instruments/{symbol}/candles"
        params = {
            "granularity": granularity,
            "count": count
        }
        headers = {
            "Authorization": f"Bearer {self.hft_model.api_key}"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            candles = data['candles']
            historical_data = pd.DataFrame([{
                'Date': candle['time'],
                'Open': float(candle['mid']['o']),
                'High': float(candle['mid']['h']),
                'Low': float(candle['mid']['l']),
                'Close': float(candle['mid']['c']),
                'Volume': candle['volume']
            } for candle in candles])
            return historical_data
        else:
            print(f"Failed to fetch historical data: {response.status_code}")
            return pd.DataFrame()

    def run(self):
        symbol = 'EUR_USD'
        take_profit_threshold = 0.001  # Example threshold for taking profit
        min_data_points = 50  # Minimum data points required for feature generation

        # Fetch additional historical data if needed
        if len(self.historical_data) < min_data_points:
            additional_data = self.fetch_additional_historical_data(symbol, 'M1', min_data_points - len(self.historical_data))
            self.historical_data = pd.concat([self.historical_data, additional_data]).reset_index(drop=True)

        while True:
            try:
                # Fetch real-time price data
                self.current_price = float(self.hft_model.get_realtime_data(symbol))
                if self.current_price is None:
                    print("No price data available. Skipping this iteration.")
                    time.sleep(60)
                    continue
                print(f"Fetched real-time price for {symbol}: {self.current_price}")
                
                # Append the current price to historical data
                new_data = pd.DataFrame([[self.current_price]], columns=['Close'])
                self.historical_data = pd.concat([self.historical_data, new_data]).reset_index(drop=True)
                
                # Ensure we have enough data points
                if len(self.historical_data) < min_data_points:
                    print("Not enough data points. Skipping this iteration.")
                    time.sleep(60)
                    continue
                
                # Generate features
                data = self.generate_features(self.historical_data)
                
                # Generate trading signal
                signal = self.model.predict(data.select_dtypes(include=[np.number]).drop(columns=['Close']).iloc[-1:])[0]
                print(f"Generated trading signal: {signal}")
                
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
        historical_data['Open'] = historical_data['Open'].astype(float)
        historical_data['High'] = historical_data['High'].astype(float)
        historical_data['Low'] = historical_data['Low'].astype(float)
        historical_data['Close'] = historical_data['Close'].astype(float)
    else:
        print(f"Warning: historical_data.csv not found at {historical_data_path}. Fetching data from OANDA.")
        historical_data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])  # Empty placeholder
    
    # Generate features and train the model
    historical_data = bot.generate_features(historical_data)
    bot.train_model(historical_data)
    
    # Run the trading bot
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Stopping the trading bot and generating report...")
        bot.generate_report()
