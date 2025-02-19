import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from typing import Dict
import os
from dotenv import load_dotenv
import urllib3
from scipy.stats import norm

# Load environment variables from .env file
load_dotenv()

# After load_dotenv()
api_key = os.getenv('NEWS_API_KEY')
print(f"News API key {'is set' if api_key else 'is not set'}")

# Create two loggers - one for suggestions and one for debug info
suggestions_logger = logging.getLogger('suggestions')
suggestions_logger.setLevel(logging.INFO)
suggestions_handler = logging.FileHandler('trading_suggestions.log', mode='w')
suggestions_handler.setFormatter(logging.Formatter('%(message)s'))
suggestions_logger.addHandler(suggestions_handler)

# Debug logger for console output
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(console_handler)

# Configure SSL and sessions
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
yf_session = requests.Session()
yf_session.verify = False

news_session = requests.Session()
news_adapter = requests.adapters.HTTPAdapter(
    max_retries=urllib3.util.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
)
news_session.mount('https://', news_adapter)
news_session.verify = False  # Disable SSL verification entirely

class MarketDataCollector:
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            debug_logger.warning("NEWS_API_KEY not set. News sentiment analysis will be disabled.")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.yf_session = yf_session
        self.news_session = news_session
        
    def get_stock_data(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        try:
            debug_logger.info(f"Fetching stock data for {ticker}")
            stock = yf.Ticker(ticker, session=self.yf_session)
            hist = stock.history(period=period)
            debug_logger.info(f"Retrieved {len(hist)} records for {ticker}")
            
            if hist.empty:
                debug_logger.error(f"No data found for ticker {ticker}")
                return pd.DataFrame()
            return hist
        except Exception as e:
            debug_logger.error(f"Failed to get ticker '{ticker}': {str(e)}")
            return pd.DataFrame()

    def get_options_chain(self, ticker: str) -> Dict:
        try:
            debug_logger.info(f"Fetching options chain for {ticker}")
            stock = yf.Ticker(ticker, session=self.yf_session)
            
            # Look for options 7-30 days out instead of next week
            min_date = datetime.now() + timedelta(days=7)
            max_date = datetime.now() + timedelta(days=30)
            options = stock.options
            
            if not options:
                debug_logger.warning(f"No options dates available for {ticker}")
                return {}
                
            debug_logger.info(f"Available options dates for {ticker}: {options}")
            
            # Find an option date between 7 and 30 days out
            valid_dates = [date for date in options 
                          if min_date <= datetime.strptime(date, '%Y-%m-%d') <= max_date]
            
            if not valid_dates:
                debug_logger.warning(f"No options found in desired date range for {ticker}")
                return {}
            
            # Use the first valid date
            exp_date = valid_dates[0]
            chain = stock.option_chain(exp_date)
            debug_logger.info(f"Retrieved options chain for {ticker} expiring {exp_date}")
            return {
                'calls': chain.calls, 
                'puts': chain.puts,
                'expiration_date': exp_date
            }
        except Exception as e:
            debug_logger.error(f"Error fetching options data for {ticker}: {str(e)}")
            return {}

    def get_market_sentiment(self, ticker: str) -> float:
        if not self.news_api_key:
            return 0.0
            
        try:
            response = self.news_session.get(
                "https://newsapi.org/v2/everything",
                headers={
                    'X-Api-Key': self.news_api_key,
                    'User-Agent': 'Mozilla/5.0'
                },
                params={
                    'q': f"{ticker} stock",
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 5
                },
                timeout=10
            )
            
            if response.status_code != 200:
                debug_logger.error(f"News API error: {response.status_code}")
                return 0.0
                
            articles = response.json().get('articles', [])
            if not articles:
                return 0.0
                
            sentiments = []
            for article in articles:
                content = f"{article.get('title', '')} {article.get('description', '')}"
                if content.strip():
                    sentiment = self.sentiment_analyzer.polarity_scores(content)
                    sentiments.append(sentiment['compound'])
                
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
            
        except Exception as e:
            debug_logger.error(f"Error fetching news sentiment for {ticker}: {str(e)}")
            return 0.0

class TradingAnalyzer:
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, periods: int = 14) -> float:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    @staticmethod
    def calculate_volatility(data: pd.DataFrame) -> float:
        return data['Close'].pct_change().std() * np.sqrt(252)

    def calculate_option_greeks(self, 
                              stock_price: float, 
                              strike_price: float, 
                              days_to_expiry: float, 
                              risk_free_rate: float = 0.05, 
                              volatility: float = None) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model
        """
        try:
            if days_to_expiry <= 0:
                return {
                    'delta': 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0
                }
            
            if volatility is None or volatility <= 0:
                volatility = 0.2  # Use default volatility if none provided or invalid
            
            T = max(days_to_expiry / 365.0, 0.01)  # Ensure minimum time value
            S = max(stock_price, 0.01)  # Ensure positive stock price
            K = max(strike_price, 0.01)  # Ensure positive strike price
            r = risk_free_rate
            sigma = volatility
            
            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Calculate Greeks
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
            theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
            vega = S*np.sqrt(T)*norm.pdf(d1)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta/365,  # Daily theta
                'vega': vega/100    # 1% volatility change
            }
        except Exception as e:
            debug_logger.error(f"Error calculating Greeks: {str(e)}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }

    def analyze_ticker(self, ticker: str, data: pd.DataFrame, options: Dict, sentiment: float) -> Dict:
        try:
            rsi = self.calculate_rsi(data)
            volatility = self.calculate_volatility(data)
            
            signals = {
                'bullish': 0,
                'bearish': 0,
                'reasons': []
            }
            
            # RSI signals - make more sensitive
            if rsi < 40:  # Was 30
                signals['bullish'] += 1
                signals['reasons'].append(f"RSI showing potential oversold ({rsi:.2f})")
            elif rsi > 60:  # Was 70
                signals['bearish'] += 1
                signals['reasons'].append(f"RSI showing potential overbought ({rsi:.2f})")
                
            # Volatility and sentiment combined signals - lower threshold
            if volatility > 0.15:  # Was 0.2
                if sentiment > 0.1:  # Was 0.2
                    signals['bullish'] += 1
                    signals['reasons'].append("Volatility with positive sentiment")
                elif sentiment < -0.1:  # Was -0.2
                    signals['bearish'] += 1
                    signals['reasons'].append("Volatility with negative sentiment")
            
            # Pure sentiment signals - lower threshold
            if abs(sentiment) > 0.15:  # Was 0.3
                if sentiment > 0:
                    signals['bullish'] += 1
                    signals['reasons'].append(f"Positive sentiment ({sentiment:.2f})")
                else:
                    signals['bearish'] += 1
                    signals['reasons'].append(f"Negative sentiment ({sentiment:.2f})")
            
            # Add trend signals
            short_trend = data['Close'].pct_change(5).mean()  # 5-day trend
            if abs(short_trend) > 0.001:  # 0.1% threshold
                if short_trend > 0:
                    signals['bullish'] += 1
                    signals['reasons'].append(f"Positive price trend ({short_trend:.2%})")
                else:
                    signals['bearish'] += 1
                    signals['reasons'].append(f"Negative price trend ({short_trend:.2%})")
            
            # Determine direction and confidence
            if signals['bullish'] == signals['bearish']:
                return {}
                
            direction = 'call' if signals['bullish'] > signals['bearish'] else 'put'
            confidence = min(100, max(signals['bullish'], signals['bearish']) * 25)  # Was 33
            
            # Select option
            chain = options.get('calls' if direction == 'call' else 'puts', pd.DataFrame())
            if chain.empty:
                return {}
                
            current_price = data['Close'].iloc[-1]
            atm_option = chain.iloc[(chain['strike'] - current_price).abs().argsort()[:1]]
            
            # Get expiration date from options data
            exp_date = options.get('expiration_date')
            if not exp_date:
                debug_logger.error(f"Could not determine expiration date for {ticker}")
                return {}
                
            # Calculate days to expiry
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            days_to_expiry = (exp_datetime - datetime.now()).days
            
            # Calculate Greeks
            greeks = self.calculate_option_greeks(
                stock_price=current_price,
                strike_price=float(atm_option['strike'].iloc[0]),
                days_to_expiry=days_to_expiry,
                volatility=volatility
            )
            
            # Add Greeks analysis to reasoning
            greek_analysis = []
            if greeks['delta'] > 0.5:
                greek_analysis.append(f"High Delta ({greeks['delta']:.2f}) indicates strong directional movement")
            if abs(greeks['theta']) > 0.1:
                greek_analysis.append(f"High Theta decay (${greeks['theta']:.2f}/day) suggests time sensitivity")
            if greeks['gamma'] > 0.05:
                greek_analysis.append(f"High Gamma ({greeks['gamma']:.2f}) indicates potential for rapid Delta changes")
            if greeks['vega'] > 1.0:
                greek_analysis.append(f"High Vega ({greeks['vega']:.2f}) shows significant volatility sensitivity")
            
            return {
                'ticker': ticker,
                'type': direction,
                'strike': float(atm_option['strike'].iloc[0]),
                'expiration': exp_date,  # Use the string date here
                'confidence': confidence,
                'reasoning': ' | '.join(signals['reasons'] + greek_analysis),
                'greeks': greeks
            }
            
        except Exception as e:
            debug_logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {}

class TradingSuggestionEngine:
    def __init__(self):
        self.data_collector = MarketDataCollector()
        self.analyzer = TradingAnalyzer()
        
    def generate_suggestion(self, ticker: str) -> Dict:
        debug_logger.info(f"Starting analysis for {ticker}")
        
        stock_data = self.data_collector.get_stock_data(ticker)
        if stock_data.empty:
            return {}
            
        options_data = self.data_collector.get_options_chain(ticker)
        if not options_data:
            return {}
            
        sentiment = self.data_collector.get_market_sentiment(ticker)
        suggestion = self.analyzer.analyze_ticker(ticker, stock_data, options_data, sentiment)
        
        return suggestion

def main():
    engine = TradingSuggestionEngine()
    tickers = ['SPY', 'QQQ']
    
    header = f"""
=== OPTIONS TRADING SUGGESTIONS ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================
RSI (Relative Strength Index): 
- Below 40: Stock may be oversold (potentially good time to buy)
- Above 60: Stock may be overbought (potentially good time to sell)
- Between 40-60: Neutral territory

Greeks Explained:
- Delta (0 to 1): Probability of option finishing in-the-money
  • 0.50: At-the-money
  • >0.70: Deep in-the-money
  • <0.30: Deep out-of-the-money
- Gamma: Rate of change in Delta (higher means faster Delta changes)
- Theta: Daily cost of time decay (negative means option loses value each day)
- Vega: Option's sensitivity to 1% change in volatility

Expiration: Days until the option expires
Confidence: How strong the trading signals are (25-100%)
====================================="""
    
    print(header)
    suggestions_logger.info(header)
    
    for ticker in tickers:
        suggestion = engine.generate_suggestion(ticker)
        if suggestion:
            # Calculate expiration days
            exp_date = datetime.strptime(suggestion['expiration'], '%Y-%m-%d')
            days_to_exp = (exp_date - datetime.now()).days
            
            output = f"\n{ticker}:\n"
            output += f"→ {suggestion['type'].upper()} @ ${suggestion['strike']:.2f}\n"
            output += f"→ Expires: {days_to_exp} days ({exp_date.strftime('%Y-%m-%d')})\n"
            output += f"→ Confidence: {suggestion['confidence']}%\n"
            output += f"→ Greeks:\n"
            output += f"  • Delta: {suggestion['greeks']['delta']:.3f} (Probability of profit, >0.50 is bullish)\n"
            output += f"  • Gamma: {suggestion['greeks']['gamma']:.3f} (Higher values mean faster Delta changes)\n"
            output += f"  • Theta: ${suggestion['greeks']['theta']:.2f}/day (Daily cost of time decay)\n"
            output += f"  • Vega: ${suggestion['greeks']['vega']:.2f} (Price change per 1% volatility change)\n"
            output += f"→ Reasoning:\n"
            
            # Add explanations for each signal
            for reason in suggestion['reasoning'].split(' | '):
                explanation = ""
                if "RSI" in reason:
                    if "overbought" in reason:
                        explanation = "(Stock may be overvalued, suggesting potential decline)"
                    elif "oversold" in reason:
                        explanation = "(Stock may be undervalued, suggesting potential rise)"
                elif "Volatility" in reason:
                    explanation = "(Price movement is significant enough to suggest a trading opportunity)"
                elif "sentiment" in reason:
                    explanation = "(News and market sentiment suggest this direction)"
                elif "trend" in reason:
                    explanation = "(Recent price movement supports this direction)"
                
                output += f"  • {reason} {explanation}\n"
            
            print(output)
            suggestions_logger.info(output)
        else:
            msg = f"\n{ticker}: No trading suggestion (signals are neutral or conflicting)"
            print(msg)
            suggestions_logger.info(msg)

if __name__ == "__main__":
    main()
