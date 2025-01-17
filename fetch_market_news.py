import requests
from datetime import datetime, timedelta

class MarketNewsFetcher:
    def __init__(self, news_api_key):
        self.news_api_key = news_api_key

    def fetch_market_news(self):
        # Using NewsAPI
        # Set the date to a recent date within the allowed range
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

# Example usage
if __name__ == "__main__":
    NEWS_API_KEY = '1d48bf7f42864457a9214e188d1e8aa1'  # Replace with your NewsAPI key
    fetcher = MarketNewsFetcher(NEWS_API_KEY)
    news_data = fetcher.fetch_market_news()
    if news_data:
        print(news_data)