# Options Trading Suggestion Engine

A Python-based trading suggestion system that analyzes ETFs (SPY, QQQ) using technical indicators, market sentiment, options Greeks, and price data to generate potential trading opportunities.

## Features

- Real-time stock data analysis using yfinance
- Technical indicators (RSI, volatility)
- News sentiment analysis using VADER
- Options chain data integration
- Black-Scholes options Greeks analysis:
  - Delta: Probability of option finishing in-the-money
  - Gamma: Rate of change in Delta
  - Theta: Time decay cost
  - Vega: Volatility sensitivity
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

The program analyzes SPY and QQQ ETFs using multiple factors:
- RSI (Relative Strength Index)
  - Below 40: Potentially oversold
  - Above 60: Potentially overbought
  - 40-60: Neutral
- Options Greeks
  - Delta (0-1): Probability of profit
    - 0.50: At-the-money
    - >0.70: Deep in-the-money
    - <0.30: Deep out-of-the-money
  - Gamma: Rate of Delta change
  - Theta: Daily time decay cost
  - Vega: Volatility sensitivity
- News sentiment
- Price trends

### Sample Output
```
=== OPTIONS TRADING SUGGESTIONS ===
Generated: 2025-02-19 07:42:30

QQQ:
→ CALL @ $539.00
→ Expires: 7 days (2025-02-27)
→ Confidence: 75%
→ Greeks:
  • Delta: 0.533 (Probability of profit, >0.50 is bullish)
  • Gamma: 0.031 (Higher values mean faster Delta changes)
  • Theta: $-0.40/day (Daily cost of time decay)
  • Vega: $0.30 (Price change per 1% volatility change)
→ Reasoning:
  • RSI showing potential overbought (69.39)
  • Positive sentiment (0.27)
  • Positive price trend (0.48%)
```

## Configuration

The system uses several configurable parameters in `trade.py`:
- RSI thresholds (default: oversold < 40, overbought > 60)
- Volatility threshold (default: 0.15)
- Sentiment thresholds (default: ±0.1 for weak, ±0.15 for strong)
- Price trend significance (default: 0.001 or 0.1%)
- Options expiration window: 7-30 days

## Limitations

- News API free tier is limited to 100 requests per day
- Options data availability depends on market hours
- Past performance does not guarantee future results
- This tool should not be used as the sole basis for trading decisions
- Greeks calculations assume Black-Scholes model assumptions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. It is not intended to be used as financial advice. Always do your own research and consult with a licensed financial advisor before making any investment decisions. Options trading involves significant risk and is not suitable for all investors.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for sentiment analysis
- [News API](https://newsapi.org/) for market news
- Black-Scholes model for options Greeks calculations
