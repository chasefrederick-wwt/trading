# Options Trading Suggestion Engine

A Python-based trading suggestion system that analyzes ETFs (SPY, QQQ) using technical indicators, market sentiment, options Greeks, and price data to generate potential trading opportunities.

## Features

- Real-time stock data analysis using yfinance
- Comprehensive Technical Analysis:
  - RSI (Relative Strength Index)
  - Moving Averages (20/50 SMA, 12/26 EMA)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Average True Range (ATR)
  - Stochastic Oscillator
  - Volume Analysis
  - Beta Calculation
  - Price Momentum
- Signal Strength Metrics:
  - Confirmation Score
  - Indicator Group Agreement
  - Signal Group Analysis:
    - Trend Indicators
    - Momentum Indicators
    - Volatility Indicators
    - Volume Indicators
- News Sentiment Analysis using VADER
- Options Analysis:
  - Black-Scholes Greeks Calculation
  - Delta: Probability of profit
  - Gamma: Rate of change in Delta
  - Theta: Time decay cost
  - Vega: Volatility sensitivity
- Detailed Trade Reasoning

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
   - Create environment file:
     - Create a file named `.env` in the project directory
     - Add your News API key:
       ```
       NEWS_API_KEY=your-api-key-here
       ```

## Usage

Run the main script:
```bash
python3 trade.py
```

The program analyzes SPY and QQQ ETFs using multiple factors:
- Technical Indicators:
  - RSI thresholds (40/60)
  - Moving Average crossovers
  - MACD signals
  - Bollinger Band positions
  - Volume analysis
  - Momentum signals
  - Stochastic crossovers
- Options Greeks Analysis
- News Sentiment
- Signal Strength Analysis

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
→ Signal Strength:
  • Confirmation Score: 75% of indicator groups agree
  • Total Signals: 8
  • Indicator Group Agreement:
    - Trend Indicators: ✓ (67% strength)
    - Momentum Indicators: ✓ (100% strength)
    - Volatility Indicators: ✗ (50% strength)
    - Volume Indicators: ✓ (100% strength)
```

## Configuration

The system uses several configurable parameters in `trade.py`:
- Technical Analysis:
  - RSI thresholds (default: oversold < 40, overbought > 60)
  - Moving Average periods (20/50 day)
  - MACD parameters (12/26/9)
  - Bollinger Bands (20-day, 2 standard deviations)
  - Volume ratio threshold (1.5x average)
  - Momentum threshold (2%)
- Options Parameters:
  - Expiration window: 7-30 days
  - Risk-free rate: 5%
  - Default volatility: 20%
- Signal Strength:
  - Confidence calculation
  - Group agreement thresholds
  - Signal confirmation weights

## Limitations

- News API free tier is limited to 100 requests per day
- Options data availability depends on market hours
- Past performance does not guarantee future results
- Technical analysis assumes historical patterns repeat
- Greeks calculations assume Black-Scholes model assumptions
- Signal strength metrics are relative indicators

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
