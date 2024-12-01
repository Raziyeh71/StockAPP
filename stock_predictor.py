from prophet import Prophet
import pandas as pd
import numpy as np

class StockPredictor:
    def __init__(self):
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            interval_width=0.95
        )
        
    def train(self, df):
        """Train the Prophet model on historical data"""
        self.model.fit(df)
        
    def predict(self, days=30):
        """Make future predictions"""
        future = self.model.make_future_dataframe(periods=days)
        forecast = self.model.predict(future)
        return forecast
    
    def get_performance_metrics(self, actual, predicted):
        """Calculate performance metrics"""
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
    def get_buy_sell_signals(self, forecast):
        """Generate buy/sell signals based on predictions"""
        signals = pd.DataFrame()
        signals['ds'] = forecast['ds']
        signals['trend'] = forecast['trend']
        signals['yhat'] = forecast['yhat']
        signals['yhat_lower'] = forecast['yhat_lower']
        signals['yhat_upper'] = forecast['yhat_upper']
        
        # Calculate trend direction
        signals['trend_direction'] = signals['trend'].diff().apply(lambda x: 1 if x > 0 else -1)
        
        # Generate signals
        signals['signal'] = signals.apply(
            lambda row: 'BUY' if row['trend_direction'] == 1 and 
                                 row['yhat'] > row['yhat_lower'] * 1.02
                       else 'SELL' if row['trend_direction'] == -1 and 
                                    row['yhat'] < row['yhat_upper'] * 0.98
                       else 'HOLD',
            axis=1
        )
        
        return signals
