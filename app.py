import streamlit as st
import plotly.graph_objects as go
from agents import StockAnalysisOrchestrator
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# Initialize the orchestrator
@st.cache_resource
def init_orchestrator():
    return StockAnalysisOrchestrator()

orchestrator = init_orchestrator()

# App title
st.title("🚀 AI-Powered Stock Predictor")
st.markdown("### Multi-Agent Stock Analysis System")

# Sidebar inputs
with st.sidebar:
    st.header("Stock Selection")
    stocks_input = st.text_area(
        "Enter stock symbols (one per line)",
        value="AAPL\nGOOG\nMSFT\nAMZN\nNVDA",
        height=150
    )
    
    analyze_button = st.button("Analyze Stocks")

# Main content
if analyze_button:
    stock_list = [s.strip() for s in stocks_input.split("\n") if s.strip()]
    
    with st.spinner("Analyzing stocks with AI agents..."):
        try:
            analysis_result = orchestrator.analyze_stocks(stock_list)
            
            if analysis_result:
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["💡 Suggestion", "📈 Prediction", "🔍 Critique"])
                
                with tab1:
                    st.header("Stock Suggestion")
                    st.markdown(f"**AI Agent's Recommendation:**")
                    st.write(analysis_result["suggestion"])
                
                with tab2:
                    st.header("Profit Prediction")
                    st.markdown(f"**Detailed Analysis:**")
                    st.write(analysis_result["prediction"])
                
                with tab3:
                    st.header("Critical Analysis")
                    st.markdown(f"**Risk Assessment:**")
                    st.write(analysis_result["critique"])
                
                # Final recommendation box
                st.success("🎯 Final Investment Recommendation")
                st.info(
                    f"""
                    Based on the multi-agent analysis:
                    1. Suggested Stock: {analysis_result['suggestion']}
                    2. Review the Prediction and Critique tabs for detailed analysis
                    3. Consider the risks before making any investment decisions
                    """
                )
            else:
                st.error("Failed to complete the analysis. Please try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with different stocks or wait a few minutes.")

else:
    st.info("👈 Enter stock symbols and click 'Analyze Stocks' to begin the analysis")
    
# Add footer with disclaimer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This is an AI-powered analysis tool for educational purposes only. 
    Always conduct your own research and consult with financial advisors before making investment decisions.
    """
)
