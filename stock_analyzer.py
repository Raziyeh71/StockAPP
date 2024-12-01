from typing import Dict, List, Tuple
from langgraph.graph import Graph, StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

class StockAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.setup_workflow()

    def setup_workflow(self):
        # Create the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("analyze_technicals", self.analyze_technical_indicators)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("make_recommendation", self.make_final_recommendation)

        # Add edges
        workflow.add_edge("analyze_technicals", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "make_recommendation")

        # Set entry point
        workflow.set_entry_point("analyze_technicals")

        # Compile
        self.graph = workflow.compile()

    def analyze_technical_indicators(self, state):
        """Analyze technical indicators using LLM"""
        template = """
        Analyze the following technical indicators for {symbol}:
        
        Price Data:
        {price_data}
        
        Forecast Data:
        {forecast_data}
        
        Provide a detailed technical analysis focusing on:
        1. Trend direction
        2. Support and resistance levels
        3. Price momentum
        4. Trading volume analysis
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        analysis = chain.invoke({
            "symbol": state.symbol,
            "price_data": state.price_data,
            "forecast_data": state.forecast_data
        })
        
        state.technical_analysis = analysis
        return state

    def analyze_sentiment(self, state):
        """Analyze market sentiment using news and social media data"""
        template = """
        Analyze the market sentiment for {symbol} based on:
        
        Recent News:
        {news_data}
        
        Company Information:
        {company_info}
        
        Previous Technical Analysis:
        {technical_analysis}
        
        Provide insights on:
        1. Overall market sentiment
        2. Key news impacts
        3. Social media sentiment
        4. Potential catalysts
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        analysis = chain.invoke({
            "symbol": state.symbol,
            "news_data": state.news_data,
            "company_info": state.company_info,
            "technical_analysis": state.technical_analysis
        })
        
        state.sentiment_analysis = analysis
        return state

    def make_final_recommendation(self, state):
        """Generate final investment recommendation"""
        template = """
        Based on the following analyses for {symbol}:
        
        Technical Analysis:
        {technical_analysis}
        
        Sentiment Analysis:
        {sentiment_analysis}
        
        Provide a comprehensive investment recommendation including:
        1. Buy/Sell/Hold recommendation
        2. Target price range
        3. Risk assessment
        4. Investment timeframe
        5. Key factors to monitor
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        recommendation = chain.invoke({
            "symbol": state.symbol,
            "technical_analysis": state.technical_analysis,
            "sentiment_analysis": state.sentiment_analysis
        })
        
        state.final_recommendation = recommendation
        return state

class GraphState:
    """State management for the analysis workflow"""
    def __init__(self, symbol, price_data, forecast_data, news_data, company_info):
        self.symbol = symbol
        self.price_data = price_data
        self.forecast_data = forecast_data
        self.news_data = news_data
        self.company_info = company_info
        self.technical_analysis = None
        self.sentiment_analysis = None
        self.final_recommendation = None
