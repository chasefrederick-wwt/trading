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
        
    def get_stock_data(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
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
            # Check if we have enough data
            if len(data) < 50:  # We need at least 50 days for the longest SMA
                debug_logger.warning(f"Insufficient data points: {len(data)}. Need at least 50.")
                return {}

            # Ensure we're working with Series objects, not single values
            close_prices = data['Close'].copy()
            high_prices = data['High'].copy()
            low_prices = data['Low'].copy()
            volume = data['Volume'].copy()
            
            # Calculate all rolling values first, keeping them as Series
            sma20_series = close_prices.rolling(window=20).mean()
            sma50_series = close_prices.rolling(window=50).mean()
            
            # MACD calculation
            ema12_series = close_prices.ewm(span=12, adjust=False).mean()
            ema26_series = close_prices.ewm(span=26, adjust=False).mean()
            
            # Only take final values after ensuring we have valid Series
            if not (sma20_series.isna().all() or sma50_series.isna().all()):
                sma_20 = sma20_series.iloc[-1]
                sma_50 = sma50_series.iloc[-1]
                ema_12 = ema12_series.iloc[-1]
                ema_26 = ema26_series.iloc[-1]
                
                # MACD components
                macd_series = ema12_series - ema26_series
                signal_series = macd_series.ewm(span=9, adjust=False).mean()
                
                macd = macd_series.iloc[-1]
                signal_line = signal_series.iloc[-1]
                macd_histogram = macd - signal_line
                
                # Bollinger Bands
                bb_middle = close_prices.rolling(window=20).mean()
                bb_std = close_prices.rolling(window=20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                
                current_price = close_prices.iloc[-1]
                
                # Ensure we have valid Bollinger Band values before calculating percent_b
                if not (bb_upper.isna().iloc[-1] or bb_lower.isna().iloc[-1]):
                    percent_b = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                else:
                    percent_b = 0.5  # Default to middle when can't calculate
                
                # ATR calculation
                high_low = high_prices - low_prices
                high_close = abs(high_prices - close_prices.shift())
                low_close = abs(low_prices - close_prices.shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1]
                
                # Volume analysis
                volume_sma = volume.rolling(window=20).mean().iloc[-1]
                volume_ratio = volume.iloc[-1] / volume_sma if volume_sma > 0 else 1.0
                
                # Beta calculation
                if '^GSPC' not in data.columns:
                    try:
                        # Convert data index to timezone-naive if it's timezone-aware
                        start_date = data.index[0].tz_localize(None) if hasattr(data.index[0], 'tz_localize') else data.index[0]
                        end_date = data.index[-1].tz_localize(None) if hasattr(data.index[-1], 'tz_localize') else data.index[-1]
                        
                        spy = yf.download('^GSPC', start=start_date, end=end_date)['Close']
                        
                        # Ensure both series have matching indexes
                        returns = close_prices.pct_change().dropna()
                        market_returns = spy.pct_change().dropna()
                        
                        # Align the series
                        returns, market_returns = returns.align(market_returns, join='inner')
                        
                        if len(returns) > 0 and len(market_returns) > 0:
                            covariance = returns.cov(market_returns)
                            market_variance = market_returns.var()
                            beta = covariance / market_variance if market_variance != 0 else 1.0
                        else:
                            beta = 1.0
                    except Exception as e:
                        debug_logger.warning(f"Error calculating beta: {str(e)}")
                        beta = 1.0
                else:
                    beta = 1.0
                
                # Momentum and Stochastic calculations
                if len(close_prices) >= 10:
                    roc = ((close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]) * 100
                else:
                    roc = 0.0
                
                # Stochastic Oscillator
                low_14 = low_prices.rolling(window=14).min()
                high_14 = high_prices.rolling(window=14).max()
                
                # Avoid division by zero in stochastic calculation
                denominator = (high_14 - low_14)
                k = pd.Series(0.0, index=close_prices.index)  # Initialize with zeros
                valid_denom = denominator != 0
                k[valid_denom] = ((close_prices - low_14) / denominator * 100)[valid_denom]
                d = k.rolling(window=3).mean()
                
                return {
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'ema_12': ema_12,
                    'ema_26': ema_26,
                    'macd': macd,
                    'macd_signal': signal_line,
                    'macd_histogram': macd_histogram,
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
            else:
                debug_logger.warning("Unable to calculate moving averages - insufficient data")
                return {}
            
        except Exception as e:
            debug_logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def analyze_ticker(self, ticker: str, data: pd.DataFrame, options: Dict, sentiment: float) -> Dict:
        try:
            debug_logger.info(f"Analyzing {ticker} with {len(data)} days of data")
            
            # Get all indicators
            rsi = self.calculate_rsi(data)
            volatility = self.calculate_volatility(data)
            indicators = self.calculate_technical_indicators(data)
            
            signals = {
                'bullish': 0,
                'bearish': 0,
                'reasons': [],
                'warnings': []
            }
            
            # Initialize signal_strength early
            signal_strength = {
                'total_signals': 0,
                'confirming_groups': 0,
                'group_agreement': {}
            }
            
            # Track signal confirmations
            signal_groups = {
                'trend_indicators': {'bullish': 0, 'bearish': 0},
                'momentum_indicators': {'bullish': 0, 'bearish': 0},
                'volatility_indicators': {'bullish': 0, 'bearish': 0},
                'volume_indicators': {'bullish': 0, 'bearish': 0}
            }
            
            # Add information about data availability
            if len(data) < 50:
                signals['warnings'].append(f"Limited historical data ({len(data)} days). Some technical indicators unavailable.")
            
            # RSI signals - make more sensitive
            if rsi < 40:  # Was 30
                signals['bullish'] += 1
                signals['reasons'].append(f"RSI showing potential oversold ({rsi:.2f})")
            elif rsi > 60:  # Was 70
                signals['bearish'] += 1
                signals['reasons'].append(f"RSI showing potential overbought ({rsi:.2f})")
            else:
                signals['reasons'].append(f"RSI is neutral ({rsi:.2f})")
            
            # Volatility and sentiment analysis
            signals['warnings'].append(f"Current volatility: {volatility:.2%}")
            signals['warnings'].append(f"Market sentiment: {sentiment:+.2f}")
            
            # Pure sentiment signals - lower threshold
            if abs(sentiment) > 0.15:  # Was 0.3
                if sentiment > 0:
                    signals['bullish'] += 1
                    signals['reasons'].append(f"Positive sentiment ({sentiment:.2f})")
                else:
                    signals['bearish'] += 1
                    signals['reasons'].append(f"Negative sentiment ({sentiment:.2f})")
            
            # Only proceed with technical analysis if we have valid indicators
            if indicators:
                # Technical Analysis Signals
                # 1. Moving Averages (Multiple timeframes)
                if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
                    signals['bullish'] += 1
                    signals['reasons'].append(f"Bullish MA crossover (20MA > 50MA)")
                elif indicators.get('sma_20', 0) < indicators.get('sma_50', 0):
                    signals['bearish'] += 1
                    signals['reasons'].append(f"Bearish MA crossover (20MA < 50MA)")
                
                # 2. MACD
                if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                    signals['bullish'] += 1
                    signals['reasons'].append(f"MACD bullish crossover ({indicators['macd']:.2f})")
                elif indicators.get('macd', 0) < indicators.get('macd_signal', 0):
                    signals['bearish'] += 1
                    signals['reasons'].append(f"MACD bearish crossover ({indicators['macd']:.2f})")
                
                # 3. Bollinger Bands
                if indicators.get('bb_percent', 0) < 0.2:
                    signals['bullish'] += 1
                    signals['reasons'].append("Price below lower Bollinger Band (oversold)")
                elif indicators.get('bb_percent', 0) > 0.8:
                    signals['bearish'] += 1
                    signals['reasons'].append("Price above upper Bollinger Band (overbought)")
                
                # 4. Volume Analysis
                if indicators.get('volume_ratio', 0) > 1.5:
                    if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                        signals['bullish'] += 2  # Strong signal when volume confirms price
                        signals['reasons'].append(f"High volume upward movement (vol ratio: {indicators['volume_ratio']:.2f})")
                    else:
                        signals['bearish'] += 2
                        signals['reasons'].append(f"High volume downward movement (vol ratio: {indicators['volume_ratio']:.2f})")
                
                # 5. Momentum
                if indicators.get('momentum', 0) > 2:  # 2% momentum threshold
                    signals['bullish'] += 1
                    signals['reasons'].append(f"Strong positive momentum ({indicators['momentum']:.2f}%)")
                elif indicators.get('momentum', 0) < -2:
                    signals['bearish'] += 1
                    signals['reasons'].append(f"Strong negative momentum ({indicators['momentum']:.2f}%)")
                
                # 6. Stochastic
                if indicators.get('stoch_k', 0) < 20 and indicators.get('stoch_k', 0) > indicators.get('stoch_d', 0):
                    signals['bullish'] += 1
                    signals['reasons'].append(f"Stochastic oversold with bullish crossover")
                elif indicators.get('stoch_k', 0) > 80 and indicators.get('stoch_k', 0) < indicators.get('stoch_d', 0):
                    signals['bearish'] += 1
                    signals['reasons'].append(f"Stochastic overbought with bearish crossover")
                
                # 7. ATR for volatility confirmation
                avg_atr_ratio = indicators.get('atr', 0) / data['Close'].iloc[-1]
                if avg_atr_ratio > 0.02:  # High volatility environment
                    if signals['bullish'] > signals['bearish']:
                        signals['bullish'] += 1
                        signals['reasons'].append(f"High volatility confirming bullish trend")
                    elif signals['bearish'] > signals['bullish']:
                        signals['bearish'] += 1
                        signals['reasons'].append(f"High volatility confirming bearish trend")
                
                # 8. Beta consideration
                if indicators.get('beta', 0) > 1.2:
                    signals['reasons'].append(f"High beta ({indicators['beta']:.2f}) suggests amplified market moves")
                elif indicators.get('beta', 0) < 0.8:
                    signals['reasons'].append(f"Low beta ({indicators['beta']:.2f}) suggests muted market moves")
            
            # Update signal groups based on indicators
            if indicators:
                # Trend indicators
                if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
                    signal_groups['trend_indicators']['bullish'] += 1
                elif indicators.get('sma_20', 0) < indicators.get('sma_50', 0):
                    signal_groups['trend_indicators']['bearish'] += 1
                
                # Momentum indicators
                if rsi < 40:
                    signal_groups['momentum_indicators']['bullish'] += 1
                elif rsi > 60:
                    signal_groups['momentum_indicators']['bearish'] += 1
                
                # Volatility indicators
                if indicators.get('bb_percent', 0) < 0.2:
                    signal_groups['volatility_indicators']['bullish'] += 1
                elif indicators.get('bb_percent', 0) > 0.8:
                    signal_groups['volatility_indicators']['bearish'] += 1
                
                # Volume indicators
                if indicators.get('volume_ratio', 0) > 1.5:
                    if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                        signal_groups['volume_indicators']['bullish'] += 1
                    else:
                        signal_groups['volume_indicators']['bearish'] += 1
            
            # Calculate signal strength metrics
            signal_strength['total_signals'] = len(signals['reasons'])
            
            # Determine overall direction
            overall_direction = 'bullish' if signals['bullish'] > signals['bearish'] else 'bearish'
            
            # Calculate group agreement
            for group_name, group_data in signal_groups.items():
                total_signals = group_data['bullish'] + group_data['bearish']
                if total_signals > 0:
                    group_direction = 'bullish' if group_data['bullish'] > group_data['bearish'] else 'bearish'
                    if group_direction == overall_direction:
                        signal_strength['confirming_groups'] += 1
                    # Fixed strength calculation
                    signal_strength['group_agreement'][group_name] = {
                        'agrees': group_direction == overall_direction,
                        'strength': max(group_data['bullish'], group_data['bearish']) / total_signals if total_signals > 0 else 0
                    }
            
            # Calculate confirmation score
            active_groups = len([g for g in signal_groups.values() if sum(g.values()) > 0])
            confirmation_score = (signal_strength['confirming_groups'] / active_groups * 100) if active_groups > 0 else 0
            
            # Adjust confidence based on signal strength and confirmation
            signal_diff = abs(signals['bullish'] - signals['bearish'])
            confidence = min(100, signal_diff * 15)  # Adjust multiplier for reasonable confidence levels
            
            # Determine direction and confidence
            if signals['bullish'] == signals['bearish']:
                return {
                    'ticker': ticker,
                    'status': 'no_suggestion',
                    'message': "No trading suggestion available",
                    'reasons': [],
                    'warnings': signals['warnings']
                }
                
            direction = 'call' if signals['bullish'] > signals['bearish'] else 'put'
            
            # Select option with more defensive checks
            chain = options.get('calls' if direction == 'call' else 'puts', pd.DataFrame())
            if chain.empty or len(chain) == 0:
                return {
                    'ticker': ticker,
                    'status': 'no_suggestion',
                    'message': "No valid options available",
                    'reasons': signals['reasons'],
                    'warnings': signals['warnings']
                }
            
            current_price = data['Close'].iloc[-1]
            
            # More defensive option selection
            try:
                strike_diffs = (chain['strike'] - current_price).abs()
                if len(strike_diffs) == 0:
                    return {
                        'ticker': ticker,
                        'status': 'no_suggestion',
                        'message': "No valid strike prices found",
                        'reasons': signals['reasons'],
                        'warnings': signals['warnings']
                    }
                
                atm_idx = strike_diffs.idxmin()
                atm_option = chain.loc[atm_idx:atm_idx]
                
                if atm_option.empty:
                    return {
                        'ticker': ticker,
                        'status': 'no_suggestion',
                        'message': "Could not find appropriate strike price",
                        'reasons': signals['reasons'],
                        'warnings': signals['warnings']
                    }
            except Exception as e:
                debug_logger.error(f"Error selecting option strike: {str(e)}")
                return {
                    'ticker': ticker,
                    'status': 'error',
                    'message': f"Error selecting option strike: {str(e)}",
                    'warnings': signals['warnings']
                }
            
            # Get expiration date from options data
            exp_date = options.get('expiration_date')
            if not exp_date:
                debug_logger.error(f"Could not determine expiration date for {ticker}")
                return {
                    'ticker': ticker,
                    'status': 'error',
                    'message': f"Could not determine expiration date for {ticker}",
                    'warnings': signals['warnings']
                }
                
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
            
            # Format the reasoning text properly
            all_reasons = []
            if signals['reasons']:
                all_reasons.extend(signals['reasons'])
            if greek_analysis:
                all_reasons.extend(greek_analysis)
            
            return {
                'ticker': ticker,
                'type': direction if 'direction' in locals() else None,
                'strike': float(atm_option['strike'].iloc[0]) if 'atm_option' in locals() else None,
                'expiration': exp_date if 'exp_date' in locals() else None,
                'confidence': confidence if 'confidence' in locals() else 0,
                'reasoning': all_reasons,  # Keep as a list instead of joining
                'warnings': signals['warnings'],
                'greeks': greeks if 'greeks' in locals() else None,
                'signal_strength': {
                    'confirmation_score': confirmation_score,
                    'total_signals': signal_strength['total_signals'],
                    'confirming_groups': signal_strength['confirming_groups'],
                    'group_details': signal_strength['group_agreement']
                }
            }
            
        except Exception as e:
            debug_logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'status': 'error',
                'message': f"Error during analysis: {str(e)}",
                'warnings': [f"Analysis failed: {str(e)}"]
            }

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
            output = f"\n{ticker}:\n"
            
            if suggestion.get('status') == 'no_suggestion':
                output += f"→ {suggestion['message']}\n"
                if suggestion.get('warnings'):
                    output += "→ Additional Information:\n"
                    for warning in suggestion['warnings']:
                        output += f"  • {warning}\n"
            elif suggestion.get('status') == 'error':
                output += f"→ Error: {suggestion['message']}\n"
            else:
                if suggestion.get('type'):
                    output += f"→ {suggestion['type'].upper()} @ ${suggestion['strike']:.2f}\n"
                    output += f"→ Expires: {suggestion['expiration']}\n"
                    output += f"→ Confidence: {suggestion['confidence']}%\n"
                
                if suggestion.get('greeks'):
                    output += f"→ Greeks:\n"
                    for greek, value in suggestion['greeks'].items():
                        output += f"  • {greek.capitalize()}: {value:.3f}\n"
                
                if suggestion.get('reasoning'):
                    output += f"→ Reasoning:\n"
                    for reason in suggestion['reasoning']:
                        output += f"  • {reason}\n"
            
            if suggestion.get('warnings'):
                output += f"→ Notes:\n"
                for warning in suggestion['warnings']:
                    output += f"  • {warning}\n"
            
            print(output)
            suggestions_logger.info(output)
        else:
            msg = f"\n{ticker}: Unable to generate trading suggestion"
            print(msg)
            suggestions_logger.info(msg)

if __name__ == "__main__":
    main()
