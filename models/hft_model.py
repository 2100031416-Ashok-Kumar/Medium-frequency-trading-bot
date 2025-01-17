import requests

class HFTModel:
    def __init__(self, api_key, account_id):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com/v3"

    def trade(self, signal, symbol):
        if signal > 0:
            self.buy(symbol)
        elif signal < 0:
            self.sell(symbol)

    def buy(self, symbol):
        # Example of buying 1000 units of EUR/USD
        data = {
            "order": {
                "units": "1000",
                "instrument": symbol,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        response = requests.post(
            f"{self.base_url}/accounts/{self.account_id}/orders",
            headers=self._headers(),
            json=data
        )
        print(f"Executed buy order for {symbol}: {response.json()}")

    def sell(self, symbol):
        # Example of selling 1000 units of EUR/USD
        data = {
            "order": {
                "units": "-1000",
                "instrument": symbol,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        response = requests.post(
            f"{self.base_url}/accounts/{self.account_id}/orders",
            headers=self._headers(),
            json=data
        )
        print(f"Executed sell order for {symbol}: {response.json()}")

    def get_realtime_data(self, symbol):
        try:
            response = requests.get(
                f"{self.base_url}/instruments/{symbol}/candles?count=1&granularity=M1",
                headers=self._headers()
            )
            data = response.json()
            print(data)  # Debugging statement to inspect the response data
            if "candles" not in data or not data["candles"]:
                print("No data returned for the symbol.")
                return None
            return float(data["candles"][-1]["mid"]["c"])  # Return the closing price of the latest candle
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            return None

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }