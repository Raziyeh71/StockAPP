import finnhub
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class StockDataFetcher:
    def __init__(self):
        self.client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

    def get_stock_data(self, symbol, period_days=365):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Convert to Unix timestamp
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        try:
            data = self.client.stock_candles(
                symbol,
                'D',
                start_timestamp,
                end_timestamp
            )
            
            if data['s'] == 'no_data':
                return None
            
            df = pd.DataFrame({
                'ds': pd.to_datetime(data['t'], unit='s'),
                'y': data['c'],  # Using closing price as target
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'volume': data['v']
            })
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_company_info(self, symbol):
        try:
            return self.client.company_profile2(symbol=symbol)
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return None

    def get_sentiment_data(self, symbol):
        try:
            news = self.client.company_news(symbol, 
                _from=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d')
            )
            return news
        except Exception as e:
            print(f"Error fetching sentiment data for {symbol}: {str(e)}")
            return []
