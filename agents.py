from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
import finnhub
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
from typing import List, Dict, Tuple, Annotated, TypedDict, Union, Any
from operator import itemgetter

load_dotenv()

class AgentState(TypedDict):
    """State for the stock analysis workflow"""
    messages: List[Union[HumanMessage, AIMessage]]
    stock_data: Dict[str, pd.DataFrame]
    current_analysis: Dict[str, Any]
    memory: Dict[str, Any]

class StockAgent:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            request_timeout=30
        )
        self.finnhub_client = finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )

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
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model=model)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a stock market expert. Analyze the provided stocks and suggest the most promising ones."),
            ("human", "Please analyze these stocks: {stock_list}. Consider market trends, technical indicators, and recent performance."),
        ])

    def analyze_market(self, state: AgentState) -> AgentState:
        """Analyze market conditions and suggest promising stocks"""
        stock_list = list(state["stock_data"].keys())
        chain = self.prompt | self.llm
        
        response = chain.invoke({"stock_list": ", ".join(stock_list)})
        state["current_analysis"]["suggestion"] = response.content
        
        # Update memory
        self.memory.save_context(
            {"input": f"Analyzed stocks: {', '.join(stock_list)}"},
            {"output": response.content}
        )
        state["memory"]["suggestion"] = self.memory.load_memory_variables({})
        
        return state

class StockPredictionAgent(StockAgent):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model=model)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a stock prediction expert. Analyze the data and provide detailed profit potential analysis."),
            ("human", "Based on the data for {symbol}, predict the profit potential and optimal timeframe."),
        ])

    def predict_performance(self, state: AgentState) -> AgentState:
        """Predict profit potential and timeframe"""
        suggestion = state["current_analysis"].get("suggestion", "")
        
        chain = self.prompt | self.llm
        response = chain.invoke({"symbol": suggestion})
        state["current_analysis"]["prediction"] = response.content
        
        # Update memory
        self.memory.save_context(
            {"input": f"Predicted performance for suggested stocks"},
            {"output": response.content}
        )
        state["memory"]["prediction"] = self.memory.load_memory_variables({})
        
        return state

class StockCriticAgent(StockAgent):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model=model)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critical analyst. Review the stock predictions and identify potential risks or oversights."),
            ("human", "Review this prediction and provide a critical analysis: {prediction}"),
        ])

    def critique_prediction(self, state: AgentState) -> AgentState:
        """Critique the prediction and identify risks"""
        prediction = state["current_analysis"].get("prediction", "")
        
        chain = self.prompt | self.llm
        response = chain.invoke({"prediction": prediction})
        state["current_analysis"]["critique"] = response.content
        
        # Update memory
        self.memory.save_context(
            {"input": "Critiqued the stock prediction"},
            {"output": response.content}
        )
        state["memory"]["critique"] = self.memory.load_memory_variables({})
        
        return state

class StockAnalysisOrchestrator:
    def __init__(self):
        self.suggestion_agent = StockSuggestionAgent()
        self.prediction_agent = StockPredictionAgent()
        self.critic_agent = StockCriticAgent()
        
        # Create the LangGraph workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("suggest", self.suggestion_agent.analyze_market)
        workflow.add_node("predict", self.prediction_agent.predict_performance)
        workflow.add_node("critique", self.critic_agent.critique_prediction)
        
        # Add edges
        workflow.add_edge('suggest', 'predict')
        workflow.add_edge('predict', 'critique')
        
        # Set entry and exit points
        workflow.set_entry_point("suggest")
        workflow.set_finish_point("critique")
        
        self.graph = workflow.compile()
    
    def analyze_stocks(self, stock_list: List[str]) -> Dict:
        """Run the full stock analysis workflow"""
        # Initialize state
        state = AgentState(
            messages=[],
            stock_data={},
            current_analysis={},
            memory={}
        )
        
        # Fetch stock data
        for symbol in stock_list:
            data = self.suggestion_agent.get_stock_data(symbol)
            if data is not None:
                state["stock_data"][symbol] = data
        
        # Run the workflow
        final_state = self.graph.invoke(state)
        
        return {
            "suggestion": final_state["current_analysis"].get("suggestion", ""),
            "prediction": final_state["current_analysis"].get("prediction", ""),
            "critique": final_state["current_analysis"].get("critique", ""),
            "memory": final_state["memory"]
        }
