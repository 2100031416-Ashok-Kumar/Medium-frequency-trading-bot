# Intelligent Multi-Factor Medium-Frequency Trading Bot

This project is an intelligent multi-factor medium-frequency trading (MFT) bot that combines machine learning models and multiple factors (technical indicators, sentiment analysis, and fundamental data) to generate trading signals and execute trades.

## Features

- Collects real-time tick data and historical OHLC data.
- Uses machine learning models for signal generation.
- Incorporates multiple factors such as technical indicators and sentiment analysis.
- Integrates a take-profit mechanism to secure profits when the market is down.
- Executes trades at a medium frequency (e.g., every minute).
- Includes a backtesting framework for testing the strategies.
- Deploys the bot with a simple dashboard for monitoring trades and performance metrics.

## Installation

1. Clone the repository.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Train the models:
    ```bash
    python trading_bot.py
    ```
2. Run the trading bot:
    ```bash
    python trading_bot.py
    ```
3. Run the dashboard:
    ```bash
    python dashboard/app.py
    ```

## Configuration

- Replace the API keys in `trading_bot.py` with your actual OANDA and NewsAPI keys.
- Ensure that the data paths for historical and tick data are correctly set.

## Technical Indicators

- Simple Moving Average (SMA)
- Relative Strength Index (RSI)

## Sentiment Analysis

- Fetches market news using NewsAPI.
- Analyzes news data to generate a sentiment score.

## Machine Learning Model

- Uses a RandomForestClassifier for signal generation.
- Trains the model on historical data with features generated from technical indicators and sentiment analysis.

## Take-Profit Mechanism

- Monitors the highest price since the position was opened.
- Sells the position if the price drops below a certain threshold from the highest price.

## License

This project is licensed under the MIT License.