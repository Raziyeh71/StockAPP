# StockAPP - Advanced AI Stock Prediction System

An advanced stock prediction system that uses multiple AI agents, LangGraph workflow, and Prophet forecasting to provide comprehensive stock market analysis.

## 🚀 Features

- Multi-Agent Analysis System:
  - Stock Suggestion Agent: Identifies promising stocks
  - Stock Prediction Agent: Forecasts performance
  - Stock Critique Agent: Provides critical evaluation
- Advanced Technologies:
  - LangGraph for agent workflow orchestration
  - Prophet for time series forecasting
  - Finnhub API for real-time market data
  - Memory-enabled agents for context retention
- Interactive Streamlit Interface

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Raziyeh71/StockAPP.git
cd StockAPP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Then edit `.env` with your API keys:
- OPENAI_API_KEY=your_openai_api_key
- FINNHUB_API_KEY=your_finnhub_api_key

## 🚀 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## 🔧 Technologies Used

- LangGraph
- LangChain
- OpenAI GPT Models
- Prophet
- Finnhub API
- Streamlit
- Pandas
- Plotly

## 📊 How It Works

1. **Stock Suggestion**: The first agent analyzes market data to identify promising stocks.
2. **Performance Prediction**: Using Prophet and AI analysis, predicts potential returns and optimal timeframes.
3. **Critical Analysis**: A dedicated agent evaluates predictions and identifies potential risks.
4. **Memory Integration**: All agents maintain context through conversation memory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
