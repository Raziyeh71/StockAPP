from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import finnhub
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
from typing import List, Dict

load_dotenv()

class StockAgent:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            request_timeout=30
        )
        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))

    def get_stock_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch stock data with retry logic"""
        for attempt in range(3):
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                data = self.finnhub_client.stock_candles(
                    symbol,
                    'D',
                    int(start_date.timestamp()),
                    int(end_date.timestamp())
                )
                if data['s'] == 'ok':
                    return pd.DataFrame({
                        'date': pd.to_datetime(data['t'], unit='s'),
                        'close': data['c'],
                        'open': data['o'],
                        'high': data['h'],
                        'low': data['l'],
                        'volume': data['v']
                    })
                time.sleep(1)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2)
        return None

class StockSuggestionAgent(StockAgent):
    def analyze_market(self, top_stocks: List[str]) -> Dict:
        """First agent that suggests most profitable stocks"""
        stocks_data = {}
        for symbol in top_stocks:
            data = self.get_stock_data(symbol)
            if data is not None:
                stocks_data[symbol] = data
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a professional stock market analyst. 
            Analyze the provided stock data and suggest which stock has the highest 
            potential for profit in the short term. Focus on recent trends, 
            momentum, and volatility."""),
            HumanMessage(content=f"Stock data: {stocks_data}")
        ])
        
        response = self.llm(analysis_prompt.format_messages())
        return {
            "suggestion": response.content,
            "data": stocks_data
        }

class StockPredictionAgent(StockAgent):
    def predict_performance(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Second agent that predicts profit potential and timeframe"""
        prediction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a stock market prediction expert.
            Analyze the stock data and provide:
            1. Expected profit percentage
            2. Optimal timeframe for the investment
            3. Key factors supporting your prediction
            Be specific with numbers and timeframes."""),
            HumanMessage(content=f"Stock: {symbol}\nData: {data.to_dict()}")
        ])
        
        response = self.llm(prediction_prompt.format_messages())
        return {
            "prediction": response.content,
            "symbol": symbol
        }

class StockCriticAgent(StockAgent):
    def critique_prediction(self, prediction: Dict, data: pd.DataFrame) -> Dict:
        """Third agent that critiques the prediction"""
        critique_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a skeptical stock market analyst.
            Your job is to critically evaluate the given prediction and identify:
            1. Potential risks and downsides
            2. Market factors that might invalidate the prediction
            3. Alternative scenarios to consider
            Be specific and data-driven in your critique."""),
            HumanMessage(content=f"Prediction: {prediction}\nData: {data.to_dict()}")
        ])
        
        response = self.llm(critique_prompt.format_messages())
        return {
            "critique": response.content,
            "symbol": prediction["symbol"]
        }

class StockAnalysisOrchestrator:
    def __init__(self):
        self.suggestion_agent = StockSuggestionAgent(model="gpt-3.5-turbo")
        self.prediction_agent = StockPredictionAgent(model="gpt-3.5-turbo")
        self.critic_agent = StockCriticAgent(model="gpt-3.5-turbo")
        
    def analyze_stocks(self, stock_list: List[str]) -> Dict:
        """Orchestrate the multi-agent analysis process"""
        try:
            # Step 1: Get stock suggestions
            suggestion_result = self.suggestion_agent.analyze_market(stock_list)
            
            # Step 2: Get detailed prediction for suggested stock
            prediction_result = self.prediction_agent.predict_performance(
                suggestion_result["suggestion"],
                suggestion_result["data"][suggestion_result["suggestion"]]
            )
            
            # Step 3: Get critique of the prediction
            critique_result = self.critic_agent.critique_prediction(
                prediction_result,
                suggestion_result["data"][suggestion_result["suggestion"]]
            )
            
            return {
                "suggestion": suggestion_result["suggestion"],
                "prediction": prediction_result["prediction"],
                "critique": critique_result["critique"]
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None
