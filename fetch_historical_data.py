import alpaca_trade_api as tradeapi
import pandas as pd
import os

API_KEY = 'PK6MK16WMS0DAANQ61IY'
API_SECRET = 'hSwyEHfpuGOiCgDaIbuz7kGLSd78m7MF8eeF3Bc1'
BASE_URL = 'https://paper-api.alpaca.markets/v2'

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Define the symbol and timeframe
symbol = 'AAPL'
timeframe = '1Min'  # Options: '1Min', '5Min', '15Min', '1D'

# Fetch tick data
start_date = '2021-01-01'
end_date = '2021-01-02'  # Fetching one day of tick data for example
bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date).df

# Convert to DataFrame
bars = bars.reset_index()
bars = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
bars.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

# Define the output directory and file path
output_dir = os.path.join(os.getcwd(), 'trading_bot_project', 'data')
output_file = os.path.join(output_dir, 'tick_data.csv')

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
bars.to_csv(output_file, index=False)

print(f"Tick data saved successfully to {output_file}.")