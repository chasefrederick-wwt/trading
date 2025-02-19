# Options Trading Suggestion Engine

A Python-based trading suggestion system that analyzes stocks using technical indicators, market sentiment, and options data to generate potential trading opportunities.

## Features

- Real-time stock data analysis using yfinance
- Technical indicators (RSI, volatility)
- News sentiment analysis using VADER
- Options chain data integration
- Options Greeks analysis (Delta, Gamma, Theta, Vega)
- Configurable trading signals
- Detailed reasoning for each suggestion

## Prerequisites

- Python 3.8+
- News API key (for sentiment analysis)
- Internet connection for real-time data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/options-trading-suggestions.git
cd options-trading-suggestions
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your News API credentials:
   - Sign up at [News API](https://newsapi.org/)
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Add your API key to `.env`:
     ```
     NEWS_API_KEY=your-api-key-here
     ```

## Usage

Run the main script:
```bash
python3 trade.py
```

The program will analyze several popular ETFs (SPY, QQQ, IWM) and generate trading suggestions based on:
- RSI (Relative Strength Index)
- Price volatility
- News sentiment
- Recent price trends

### Sample Output
```
=== OPTIONS TRADING SUGGESTIONS ===
Generated: 2025-02-19 07:25:07
=====================================

QQQ:
→ CALL @ $539.00
→ Expires: 42 days (2025-04-02)
→ Confidence: 75%
→ Reasoning:
  • RSI showing potential overbought (69.39) (Stock may be overvalued, suggesting potential decline)
  • Volatility with positive sentiment (Price movement is significant enough to suggest a trading opportunity)
  • Positive sentiment (0.27) (News and market sentiment suggest this direction)
  • Positive price trend (0.48%) (Recent price movement supports this direction)
```

## Configuration

The system uses several configurable parameters that can be adjusted in `trade.py`:
- RSI thresholds (default: oversold < 40, overbought > 60)
- Volatility threshold (default: 0.15)
- Sentiment thresholds (default: ±0.1 for weak, ±0.15 for strong)
- Price trend significance (default: 0.001 or 0.1%)

## Limitations

- News API free tier is limited to 100 requests per day
- Options data availability depends on market hours
- Past performance does not guarantee future results
- This tool should not be used as the sole basis for trading decisions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. It is not intended to be used as financial advice. Always do your own research and consult with a licensed financial advisor before making any investment decisions.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [News API](https://newsapi.org/) for market news
