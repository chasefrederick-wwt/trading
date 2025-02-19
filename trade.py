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

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various technical indicators for analysis"""
        try:
            # Prepare data
            close_prices = data['Close']
            high_prices = data['High']
            low_prices = data['Low']
            volume = data['Volume']
            
            # Moving Averages
            sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
            ema_12 = close_prices.ewm(span=12).mean().iloc[-1]
            ema_26 = close_prices.ewm(span=26).mean().iloc[-1]
            
            # MACD
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean().iloc[-1]
            macd_histogram = macd_line - signal_line
            
            # Bollinger Bands (20-day, 2 standard deviations)
            bb_middle = close_prices.rolling(window=20).mean()
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Calculate %B (position within Bollinger Bands)
            current_price = close_prices.iloc[-1]
            percent_b = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Average True Range (ATR)
            high_low = high_prices - low_prices
            high_close = abs(high_prices - close_prices.shift())
            low_close = abs(low_prices - close_prices.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Volume analysis
            volume_sma = volume.rolling(window=20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume_sma
            
            # Beta calculation (using SPY as market proxy if not already SPY)
            if '^GSPC' not in data.columns:
                spy = yf.download('^GSPC', start=data.index[0], end=data.index[-1])['Close']
                returns = close_prices.pct_change()
                market_returns = spy.pct_change()
                covariance = returns.cov(market_returns)
                market_variance = market_returns.var()
                beta = covariance / market_variance
            else:
                beta = 1.0  # If analyzing SPY itself
            
            # Momentum indicators
            roc = ((close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]) * 100
            
            # Stochastic Oscillator
            low_14 = low_prices.rolling(window=14).min()
            high_14 = high_prices.rolling(window=14).max()
            k = ((close_prices - low_14) / (high_14 - low_14)) * 100
            d = k.rolling(window=3).mean()
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'macd': macd_line.iloc[-1],
                'macd_signal': signal_line,
                'macd_histogram': macd_histogram.iloc[-1],
                'bb_upper': bb_upper.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'bb_percent': percent_b,
                'atr': atr,
                'volume_ratio': volume_ratio,
                'beta': beta,
                'momentum': roc,
                'stoch_k': k.iloc[-1],
                'stoch_d': d.iloc[-1]
            }
        except Exception as e:
            debug_logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def analyze_ticker(self, ticker: str, data: pd.DataFrame, options: Dict, sentiment: float) -> Dict:
        try:
            # Get all indicators
            rsi = self.calculate_rsi(data)
            volatility = self.calculate_volatility(data)
            indicators = self.calculate_technical_indicators(data)
            
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
            
            # Technical Analysis Signals
            
            # 1. Moving Averages (Multiple timeframes)
            if indicators['sma_20'] > indicators['sma_50']:
                signals['bullish'] += 1
                signals['reasons'].append(f"Bullish MA crossover (20MA > 50MA)")
            elif indicators['sma_20'] < indicators['sma_50']:
                signals['bearish'] += 1
                signals['reasons'].append(f"Bearish MA crossover (20MA < 50MA)")
            
            # 2. MACD
            if indicators['macd'] > indicators['macd_signal']:
                signals['bullish'] += 1
                signals['reasons'].append(f"MACD bullish crossover ({indicators['macd']:.2f})")
            elif indicators['macd'] < indicators['macd_signal']:
                signals['bearish'] += 1
                signals['reasons'].append(f"MACD bearish crossover ({indicators['macd']:.2f})")
            
            # 3. Bollinger Bands
            if indicators['bb_percent'] < 0.2:
                signals['bullish'] += 1
                signals['reasons'].append("Price below lower Bollinger Band (oversold)")
            elif indicators['bb_percent'] > 0.8:
                signals['bearish'] += 1
                signals['reasons'].append("Price above upper Bollinger Band (overbought)")
            
            # 4. Volume Analysis
            if indicators['volume_ratio'] > 1.5:
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    signals['bullish'] += 2  # Strong signal when volume confirms price
                    signals['reasons'].append(f"High volume upward movement (vol ratio: {indicators['volume_ratio']:.2f})")
                else:
                    signals['bearish'] += 2
                    signals['reasons'].append(f"High volume downward movement (vol ratio: {indicators['volume_ratio']:.2f})")
            
            # 5. Momentum
            if indicators['momentum'] > 2:  # 2% momentum threshold
                signals['bullish'] += 1
                signals['reasons'].append(f"Strong positive momentum ({indicators['momentum']:.2f}%)")
            elif indicators['momentum'] < -2:
                signals['bearish'] += 1
                signals['reasons'].append(f"Strong negative momentum ({indicators['momentum']:.2f}%)")
            
            # 6. Stochastic
            if indicators['stoch_k'] < 20 and indicators['stoch_k'] > indicators['stoch_d']:
                signals['bullish'] += 1
                signals['reasons'].append(f"Stochastic oversold with bullish crossover")
            elif indicators['stoch_k'] > 80 and indicators['stoch_k'] < indicators['stoch_d']:
                signals['bearish'] += 1
                signals['reasons'].append(f"Stochastic overbought with bearish crossover")
            
            # 7. ATR for volatility confirmation
            avg_atr_ratio = indicators['atr'] / data['Close'].iloc[-1]
            if avg_atr_ratio > 0.02:  # High volatility environment
                if signals['bullish'] > signals['bearish']:
                    signals['bullish'] += 1
                    signals['reasons'].append(f"High volatility confirming bullish trend")
                elif signals['bearish'] > signals['bullish']:
                    signals['bearish'] += 1
                    signals['reasons'].append(f"High volatility confirming bearish trend")
            
            # 8. Beta consideration
            if indicators['beta'] > 1.2:
                signals['reasons'].append(f"High beta ({indicators['beta']:.2f}) suggests amplified market moves")
            elif indicators['beta'] < 0.8:
                signals['reasons'].append(f"Low beta ({indicators['beta']:.2f}) suggests muted market moves")
            
            # Adjust confidence based on signal strength and confirmation
            signal_diff = abs(signals['bullish'] - signals['bearish'])
            confidence = min(100, signal_diff * 15)  # Adjust multiplier for reasonable confidence levels
            
            # Determine direction and confidence
            if signals['bullish'] == signals['bearish']:
                return {}
                
            direction = 'call' if signals['bullish'] > signals['bearish'] else 'put'
            
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
            
            # Track signal confirmations
            signal_groups = {
                'trend_indicators': {
                    'bullish': 0,
                    'bearish': 0,
                    'signals': ['sma_crossover', 'macd', 'momentum']
                },
                'momentum_indicators': {
                    'bullish': 0,
                    'bearish': 0,
                    'signals': ['rsi', 'stochastic']
                },
                'volatility_indicators': {
                    'bullish': 0,
                    'bearish': 0,
                    'signals': ['bollinger_bands', 'atr']
                },
                'volume_indicators': {
                    'bullish': 0,
                    'bearish': 0,
                    'signals': ['volume_ratio']
                }
            }
            
            for group, data in signal_groups.items():
                if data['signals'][0] in signals['reasons'] or data['signals'][1] in signals['reasons']:
                    if data['signals'][0] in signals['reasons']:
                        data['bullish'] += 1
                    if data['signals'][1] in signals['reasons']:
                        data['bearish'] += 1
            
            # Calculate signal strength metrics
            signal_strength = {
                'total_signals': len(signals['reasons']),
                'confirming_groups': 0,
                'group_agreement': {}
            }
            
            for group, data in signal_groups.items():
                if data['bullish'] > 0 or data['bearish'] > 0:
                    # Calculate group direction
                    group_direction = 'bullish' if data['bullish'] > data['bearish'] else 'bearish'
                    overall_direction = 'bullish' if signals['bullish'] > signals['bearish'] else 'bearish'
                    
                    # Check if group agrees with overall direction
                    if group_direction == overall_direction:
                        signal_strength['confirming_groups'] += 1
                    
                    # Store group agreement details
                    signal_strength['group_agreement'][group] = {
                        'agrees': group_direction == overall_direction,
                        'strength': max(data['bullish'], data['bearish']) / len(data['signals'])
                    }
            
            # Calculate overall confirmation percentage
            confirmation_score = (signal_strength['confirming_groups'] / len(signal_groups)) * 100
            
            # Add signal strength to the output
            return {
                'ticker': ticker,
                'type': direction,
                'strike': float(atm_option['strike'].iloc[0]),
                'expiration': exp_date,
                'confidence': confidence,
                'reasoning': ' | '.join(signals['reasons'] + greek_analysis),
                'greeks': greeks,
                'signal_strength': {
                    'confirmation_score': confirmation_score,
                    'total_signals': signal_strength['total_signals'],
                    'confirming_groups': signal_strength['confirming_groups'],
                    'group_details': signal_strength['group_agreement']
                }
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
            
            output += f"→ Signal Strength:\n"
            output += f"  • Confirmation Score: {suggestion['signal_strength']['confirmation_score']:.0f}% of indicator groups agree\n"
            output += f"  • Total Signals: {suggestion['signal_strength']['total_signals']}\n"
            output += f"  • Indicator Group Agreement:\n"
            for group, details in suggestion['signal_strength']['group_details'].items():
                agreement = "✓" if details['agrees'] else "✗"
                output += f"    - {group.replace('_', ' ').title()}: {agreement} ({details['strength']*100:.0f}% strength)\n"
            
            print(output)
            suggestions_logger.info(output)
        else:
            msg = f"\n{ticker}: No trading suggestion (signals are neutral or conflicting)"
            print(msg)
            suggestions_logger.info(msg)

if __name__ == "__main__":
    main()
