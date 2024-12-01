# AI Stock Predictor

A cutting-edge stock prediction system that combines LangGraph, Prophet, and advanced AI analysis to provide intelligent stock trading recommendations.

## Features

- Real-time stock data fetching using Finnhub API
- Advanced time series forecasting with Facebook Prophet
- Intelligent market analysis using LangGraph and GPT-4
- Technical and sentiment analysis
- Interactive web interface with Streamlit
- Trading signals generation
- Comprehensive investment recommendations

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter a stock symbol (e.g., AAPL, GOOGL)
2. Adjust the prediction timeframe
3. Click "Analyze Stock"
4. View predictions, trading signals, and AI analysis

## Components

- `stock_data.py`: Handles data fetching from Finnhub
- `stock_predictor.py`: Implements Prophet-based prediction
- `stock_analyzer.py`: Contains LangGraph workflow for analysis
- `app.py`: Streamlit web interface

## Requirements

- Python 3.8+
- OpenAI API key
- Finnhub API key

## License

MIT
