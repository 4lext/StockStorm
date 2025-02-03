import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.data.storage.data_manager import FinancialDataStorage
from src.analysis.financial_analyst import FinancialAnalystAI
import os
from dotenv import load_dotenv
import inspect
import requests
from src.data.fundamentals.fundamentals_manager import FundamentalAnalyzer

load_dotenv()  # Load .env for API keys

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(df, period=14):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(df, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return sma, upper_band, lower_band

class CandlestickPatterns:
    @staticmethod
    def doji(df, threshold=0.1):
        """
        Detect Doji patterns with strength measurement
        Returns: Series with values:
        0: No Doji
        1: Weak Doji (body is between 7-10% of range)
        2: Strong Doji (body is less than 7% of range)
        """
        try:
            body = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            
            # Calculate average daily range for context
            adr = total_range.rolling(window=10).mean()
            
            # Calculate body to range ratio
            body_range_ratio = body / total_range.where(total_range != 0, np.inf)
            
            # Detect Doji patterns with strength
            strong_doji = (body_range_ratio <= 0.07) & (total_range >= adr * 0.5)
            weak_doji = (body_range_ratio <= threshold) & (body_range_ratio > 0.07) & (total_range > 0)
            
            return np.where(strong_doji, 2, np.where(weak_doji, 1, 0))
        except Exception as e:
            print(f"Error in doji detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def hammer(df, body_threshold=0.3, shadow_threshold=2):
        """Detect Hammer patterns"""
        try:
            body = abs(df['close'] - df['open'])
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            total_range = df['high'] - df['low']
            
            is_hammer = (
                (body > 0) &  # Must have a body
                (lower_shadow >= body * shadow_threshold) &  # Long lower shadow
                (upper_shadow <= body * 0.1) &  # Very small upper shadow
                (body <= total_range * body_threshold)  # Body is small relative to total range
            )
            
            return np.where(is_hammer & (df['close'] > df['open']), 1,  # Bullish hammer
                          np.where(is_hammer & (df['close'] < df['open']), -1, 0))  # Bearish hammer
        except Exception as e:
            print(f"Error in hammer detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def engulfing(df):
        """
        Detect Bullish and Bearish Engulfing patterns
        Returns:
        1: Bullish Engulfing
        -1: Bearish Engulfing
        0: No Engulfing pattern
        """
        try:
            prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
            curr_body = abs(df['close'] - df['open'])
            
            bullish = (
                (df['close'] > df['open']) &  # Current candle is bullish
                (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
                (df['close'] > df['open'].shift(1)) &  # Current close higher than previous open
                (df['open'] < df['close'].shift(1)) &  # Current open lower than previous close
                (curr_body > prev_body)  # Current body engulfs previous body
            )
            
            bearish = (
                (df['close'] < df['open']) &  # Current candle is bearish
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
                (df['close'] < df['open'].shift(1)) &  # Current close lower than previous open
                (df['open'] > df['close'].shift(1)) &  # Current open higher than previous close
                (curr_body > prev_body)  # Current body engulfs previous body
            )
            
            return np.where(bullish, 1, np.where(bearish, -1, 0))
        except Exception as e:
            print(f"Error in engulfing detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def shooting_star(df, body_threshold=0.3, shadow_threshold=2):
        """Detect Shooting Star patterns"""
        try:
            body = abs(df['close'] - df['open'])
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
            total_range = df['high'] - df['low']
            
            is_shooting_star = (
                (body > 0) &  # Must have a body
                (upper_shadow >= body * shadow_threshold) &  # Long upper shadow
                (lower_shadow <= body * 0.1) &  # Very small lower shadow
                (body <= total_range * body_threshold)  # Body is small relative to total range
            )
            
            return np.where(is_shooting_star & (df['close'] < df['open']), -1,  # Bearish shooting star
                          np.where(is_shooting_star & (df['close'] > df['open']), 1, 0))  # Bullish shooting star
        except Exception as e:
            print(f"Error in shooting star detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def morning_star(df, body_threshold=0.3):
        """Detect Morning Star patterns"""
        try:
            # First day: long bearish candle
            first_bearish = (df['close'].shift(2) < df['open'].shift(2)) & \
                          (abs(df['close'].shift(2) - df['open'].shift(2)) > \
                           (df['high'].shift(2) - df['low'].shift(2)) * body_threshold)
            
            # Second day: small body
            second_small_body = abs(df['close'].shift(1) - df['open'].shift(1)) < \
                              abs(df['close'].shift(2) - df['open'].shift(2)) * 0.3
            
            # Third day: long bullish candle
            third_bullish = (df['close'] > df['open']) & \
                          (abs(df['close'] - df['open']) > (df['high'] - df['low']) * body_threshold) & \
                          (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)
            
            morning_star = first_bearish & second_small_body & third_bullish
            return np.where(morning_star, 1, 0)
        except Exception as e:
            print(f"Error in morning star detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def evening_star(df, body_threshold=0.3):
        """Detect Evening Star patterns"""
        try:
            # First day: long bullish candle
            first_bullish = (df['close'].shift(2) > df['open'].shift(2)) & \
                          (abs(df['close'].shift(2) - df['open'].shift(2)) > \
                           (df['high'].shift(2) - df['low'].shift(2)) * body_threshold)
            
            # Second day: small body
            second_small_body = abs(df['close'].shift(1) - df['open'].shift(1)) < \
                              abs(df['close'].shift(2) - df['open'].shift(2)) * 0.3
            
            # Third day: long bearish candle
            third_bearish = (df['close'] < df['open']) & \
                          (abs(df['close'] - df['open']) > (df['high'] - df['low']) * body_threshold) & \
                          (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)
            
            evening_star = first_bullish & second_small_body & third_bearish
            return np.where(evening_star, -1, 0)
        except Exception as e:
            print(f"Error in evening star detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def three_white_soldiers(df, body_threshold=0.7):
        """Detect Three White Soldiers pattern"""
        try:
            # All three candles should be bullish
            bullish_candles = (df['close'] > df['open']) & \
                            (df['close'].shift(1) > df['open'].shift(1)) & \
                            (df['close'].shift(2) > df['open'].shift(2))
            
            # Each candle should have a substantial body
            strong_bodies = (abs(df['close'] - df['open']) > (df['high'] - df['low']) * body_threshold) & \
                          (abs(df['close'].shift(1) - df['open'].shift(1)) > \
                           (df['high'].shift(1) - df['low'].shift(1)) * body_threshold) & \
                          (abs(df['close'].shift(2) - df['open'].shift(2)) > \
                           (df['high'].shift(2) - df['low'].shift(2)) * body_threshold)
            
            # Each close should be higher than the previous close
            higher_closes = (df['close'] > df['close'].shift(1)) & \
                          (df['close'].shift(1) > df['close'].shift(2))
            
            three_soldiers = bullish_candles & strong_bodies & higher_closes
            return np.where(three_soldiers, 1, 0)
        except Exception as e:
            print(f"Error in three white soldiers detection: {str(e)}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def three_black_crows(df, body_threshold=0.7):
        """Detect Three Black Crows pattern"""
        try:
            # All three candles should be bearish
            bearish_candles = (df['close'] < df['open']) & \
                            (df['close'].shift(1) < df['open'].shift(1)) & \
                            (df['close'].shift(2) < df['open'].shift(2))
            
            # Each candle should have a substantial body
            strong_bodies = (abs(df['close'] - df['open']) > (df['high'] - df['low']) * body_threshold) & \
                          (abs(df['close'].shift(1) - df['open'].shift(1)) > \
                           (df['high'].shift(1) - df['low'].shift(1)) * body_threshold) & \
                          (abs(df['close'].shift(2) - df['open'].shift(2)) > \
                           (df['high'].shift(2) - df['low'].shift(2)) * body_threshold)
            
            # Each close should be lower than the previous close
            lower_closes = (df['close'] < df['close'].shift(1)) & \
                          (df['close'].shift(1) < df['close'].shift(2))
            
            three_crows = bearish_candles & strong_bodies & lower_closes
            return np.where(three_crows, -1, 0)
        except Exception as e:
            print(f"Error in three black crows detection: {str(e)}")
            return pd.Series(0, index=df.index)

class AlpacaCandlestickAnalyzer:
    def __init__(self, api_key, secret_key):
        """Initialize with Alpaca API credentials"""
        self.client = StockHistoricalDataClient(api_key, secret_key)
        self.storage = FinancialDataStorage()
        self.analyst = FinancialAnalystAI(
            os.getenv("ANTHROPIC_API_KEY")
        )
        self.symbol = None
        self.fundamental_api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
        self.fundamental_base_url = "https://api.financialdatasets.ai/financials/"
        
        # Set up requests session
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.fundamental_api_key,
            "Accept": "application/json"
        })

    def get_stock_data(self, symbol, start_date, end_date=None):
        """Fetch stock data from Alpaca"""
        self.symbol = symbol  # Store the symbol when fetching data
        
        if end_date is None:
            end_date = datetime.now()
            
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date)
        )
        
        try:
            bars = self.client.get_stock_bars(request_params)
            df = bars.df
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            
            return df
        
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def analyze_patterns(self, df):
        """Analyze candlestick patterns and add technical indicators"""
        if df is None or df.empty:
            print("No data to analyze")
            return None
        
        df = df.copy()
        required_columns = ['open', 'close', 'high', 'low']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing columns: {missing}")
            return None
        
        # Calculate candlestick patterns
        df['DOJI'] = CandlestickPatterns.doji(df)
        df['HAMMER'] = CandlestickPatterns.hammer(df)
        df['ENGULFING'] = CandlestickPatterns.engulfing(df)
        df['SHOOTING_STAR'] = CandlestickPatterns.shooting_star(df)
        df['MORNING_STAR'] = CandlestickPatterns.morning_star(df)
        df['EVENING_STAR'] = CandlestickPatterns.evening_star(df)
        df['THREE_WHITE_SOLDIERS'] = CandlestickPatterns.three_white_soldiers(df)
        df['THREE_BLACK_CROWS'] = CandlestickPatterns.three_black_crows(df)
        
        # Calculate technical indicators
        df['RSI'] = TechnicalIndicators.calculate_rsi(df)
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = TechnicalIndicators.calculate_macd(df)
        df['SMA20'], df['BB_UPPER'], df['BB_LOWER'] = TechnicalIndicators.calculate_bollinger_bands(df)
        
        return df
    
    def get_active_patterns(self, df, date=None):
        """Get active patterns for a specific date"""
        if df is None or df.empty:
            return {}
            
        if date is None:
            date = df.index[-1]
            
        patterns = {
            'DOJI': 'Doji',
            'HAMMER': 'Hammer',
            'ENGULFING': 'Engulfing',
            'SHOOTING_STAR': 'Shooting Star',
            'MORNING_STAR': 'Morning Star',
            'EVENING_STAR': 'Evening Star',
            'THREE_WHITE_SOLDIERS': 'Three White Soldiers',
            'THREE_BLACK_CROWS': 'Three Black Crows'
        }
        
        active_patterns = {}
        try:
            for col, pattern_name in patterns.items():
                value = df.loc[date, col]
                if value != 0:
                    if col == 'DOJI':
                        strength = 'Strong' if value == 2 else 'Weak'
                        active_patterns[pattern_name] = f'Neutral ({strength})'
                    else:
                        active_patterns[pattern_name] = 'Bullish' if value > 0 else 'Bearish'
        except KeyError:
            print(f"Date {date} not found in data")
            return {}
            
        return active_patterns

class MarketAnalysis:
    @staticmethod
    def analyze_market_conditions(df):
        """Analyze overall market conditions"""
        if df is None or len(df) < 50:  # Need at least 50 days for reliable analysis
            return None
            
        try:
            conditions = {
                'trend': {},
                'momentum': {},
                'volatility': {},
                'support_resistance': {}
            }
            
            # Trend Analysis
            sma20 = df['close'].rolling(window=20).mean()
            sma50 = df['close'].rolling(window=50).mean()
            
            if len(df) >= 20:  # Check if we have enough data for 20-day analysis
                current_price = df['close'].iloc[-1]
                conditions['trend'] = {
                    'price_vs_sma20': 'above' if current_price > sma20.iloc[-1] else 'below',
                    'price_vs_sma50': 'above' if current_price > sma50.iloc[-1] else 'below' if len(df) >= 50 else 'insufficient data',
                    'sma20_vs_sma50': 'bullish' if sma20.iloc[-1] > sma50.iloc[-1] else 'bearish' if len(df) >= 50 else 'insufficient data',
                    'short_term': 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-5] else 'bearish' if len(df) >= 5 else 'insufficient data',
                    'medium_term': 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'bearish'
                }
            else:
                conditions['trend'] = {
                    'price_vs_sma20': 'insufficient data',
                    'price_vs_sma50': 'insufficient data',
                    'sma20_vs_sma50': 'insufficient data',
                    'short_term': 'insufficient data',
                    'medium_term': 'insufficient data'
                }
            
            # Momentum Analysis
            if len(df) >= 14:  # Need at least 14 days for RSI
                rsi = TechnicalIndicators.calculate_rsi(df)
                macd, signal, _ = TechnicalIndicators.calculate_macd(df)
                
                conditions['momentum'] = {
                    'rsi': rsi.iloc[-1],
                    'rsi_condition': 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral',
                    'macd_signal': 'bullish' if macd.iloc[-1] > signal.iloc[-1] else 'bearish',
                    'macd_trend': 'strengthening' if macd.iloc[-1] > macd.iloc[-5] else 'weakening' if len(df) >= 5 else 'insufficient data'
                }
            else:
                conditions['momentum'] = {
                    'rsi': None,
                    'rsi_condition': 'insufficient data',
                    'macd_signal': 'insufficient data',
                    'macd_trend': 'insufficient data'
                }
            
            # Volatility Analysis
            if len(df) >= 20:
                _, upper_bb, lower_bb = TechnicalIndicators.calculate_bollinger_bands(df)
                atr = df['high'] - df['low']
                
                conditions['volatility'] = {
                    'bb_width': (upper_bb.iloc[-1] - lower_bb.iloc[-1]) / df['close'].iloc[-1],
                    'atr': atr.iloc[-1],
                    'avg_atr': atr.rolling(window=14).mean().iloc[-1] if len(df) >= 14 else None,
                    'volatility_state': 'high' if atr.iloc[-1] > atr.rolling(window=14).mean().iloc[-1] else 'low' if len(df) >= 14 else 'insufficient data'
                }
            else:
                conditions['volatility'] = {
                    'bb_width': None,
                    'atr': None,
                    'avg_atr': None,
                    'volatility_state': 'insufficient data'
                }
            
            # Support and Resistance Analysis
            if len(df) >= 20:
                price_history = df['close'].tail(20)
                support = price_history.min()
                resistance = price_history.max()
                current_price = df['close'].iloc[-1]
                
                conditions['support_resistance'] = {
                    'nearest_support': support,
                    'nearest_resistance': resistance,
                    'distance_to_support': ((current_price - support) / current_price) * 100,
                    'distance_to_resistance': ((resistance - current_price) / current_price) * 100
                }
            else:
                conditions['support_resistance'] = {
                    'nearest_support': None,
                    'nearest_resistance': None,
                    'distance_to_support': None,
                    'distance_to_resistance': None
                }
            
            return conditions
            
        except Exception as e:
            print(f"Error in market analysis: {str(e)}")
            return None
    
    @staticmethod
    def generate_market_summary(conditions):
        """Generate a readable market summary"""
        if conditions is None:
            return "Insufficient data for market analysis"
            
        summary = []
        
        # Trend Summary
        trend = conditions['trend']
        summary.append("TREND ANALYSIS:")
        if trend['short_term'] != 'insufficient data':
            summary.append(f"- Short-term trend is {trend['short_term']}")
        if trend['medium_term'] != 'insufficient data':
            summary.append(f"- Medium-term trend is {trend['medium_term']}")
        if trend['price_vs_sma20'] != 'insufficient data':
            summary.append(f"- Price is {trend['price_vs_sma20']} 20-day SMA")
        if trend['price_vs_sma50'] != 'insufficient data':
            summary.append(f"and {trend['price_vs_sma50']} 50-day SMA")
        if trend['sma20_vs_sma50'] != 'insufficient data':
            summary.append(f"- Moving averages show {trend['sma20_vs_sma50']} alignment")
        
        # Momentum Summary
        momentum = conditions['momentum']
        if momentum['rsi'] is not None:
            summary.append("\nMOMENTUM ANALYSIS:")
            summary.append(f"- RSI is at {momentum['rsi']:.2f} ({momentum['rsi_condition']})")
            if momentum['macd_signal'] != 'insufficient data':
                summary.append(f"- MACD shows {momentum['macd_signal']} momentum and is {momentum['macd_trend']}")
        
        # Volatility Summary
        volatility = conditions['volatility']
        if volatility['atr'] is not None:
            summary.append("\nVOLATILITY ANALYSIS:")
            summary.append(f"- Current volatility is {volatility['volatility_state']}")
            if volatility['avg_atr'] is not None:
                summary.append(f"- ATR: {volatility['atr']:.2f} (14-day avg: {volatility['avg_atr']:.2f})")
        
        # Support/Resistance Summary
        sr = conditions['support_resistance']
        if sr['nearest_support'] is not None:
            summary.append("\nSUPPORT/RESISTANCE ANALYSIS:")
            summary.append(f"- Nearest support: ${sr['nearest_support']:.2f} ({sr['distance_to_support']:.1f}% below)")
            summary.append(f"- Nearest resistance: ${sr['nearest_resistance']:.2f} ({sr['distance_to_resistance']:.1f}% above)")
        
        return "\n".join(summary)

def plot_candlestick(df, symbol):
    """Plot and save candlestick chart with detected patterns and technical indicators"""
    if df is None or df.empty:
        print("No data to plot")
        return
    
    # Set matplotlib backend to Agg to prevent GUI issues
    import matplotlib
    matplotlib.use('Agg')  # Add this line first
    import matplotlib.pyplot as plt
    
    # Calculate technical indicators
    rsi = TechnicalIndicators.calculate_rsi(df)
    macd, signal, histogram = TechnicalIndicators.calculate_macd(df)
    sma20, upper_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(df, period=20)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 14))  # Slightly taller for fundamentals
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # Main candlestick chart
    ax1 = fig.add_subplot(gs[0])
    
    # Calculate price range for better y-axis limits
    price_range = df['high'].max() - df['low'].min()
    y_margin = price_range * 0.05
   
   
    # Plot candlesticks
    for idx, (date, row) in enumerate(df.iterrows()):
        # Plot candlestick body
        color = 'green' if row['close'] >= row['open'] else 'red'
        bottom = min(row['open'], row['close'])
        height = abs(row['close'] - row['open'])
        
        # Add body
        rect = plt.Rectangle((idx, bottom), 0.8, height, 
                           facecolor=color, edgecolor='black', alpha=0.7)
        ax1.add_patch(rect)
        
        # Add shadows
        ax1.plot([idx + 0.4, idx + 0.4], [row['low'], bottom], color='black')
        ax1.plot([idx + 0.4, idx + 0.4], [row['high'], bottom + height], color='black')
        
        # Add pattern markers
        marker_y = row['high'] + (price_range * 0.02)
        if row['DOJI'] > 0:
            color = 'blue' if row['DOJI'] == 2 else 'cyan'
            ax1.scatter(idx + 0.4, marker_y, color=color, marker='o', s=100)
        
        if row['HAMMER'] != 0:
            color = 'green' if row['HAMMER'] > 0 else 'red'
            ax1.scatter(idx + 0.4, marker_y, color=color, marker='^', s=100)
        
        if row['ENGULFING'] != 0:
            color = 'green' if row['ENGULFING'] > 0 else 'red'
            ax1.scatter(idx + 0.4, marker_y, color=color, marker='s', s=100)
        
        if row['SHOOTING_STAR'] != 0:
            color = 'red' if row['SHOOTING_STAR'] < 0 else 'green'
            ax1.scatter(idx + 0.4, marker_y, color=color, marker='v', s=100)
        
        if row['MORNING_STAR'] != 0 or row['EVENING_STAR'] != 0:
            color = 'green' if row['MORNING_STAR'] > 0 else 'red'
            ax1.scatter(idx + 0.4, marker_y, color=color, marker='*', s=150)
        
        if row['THREE_WHITE_SOLDIERS'] != 0 or row['THREE_BLACK_CROWS'] != 0:
            color = 'green' if row['THREE_WHITE_SOLDIERS'] > 0 else 'red'
            ax1.scatter(idx + 0.4, marker_y, color=color, marker='D', s=100)
    
    # Plot Bollinger Bands
    ax1.plot(range(len(df)), sma20, 'b--', alpha=0.6, label='20-day SMA')
    ax1.plot(range(len(df)), upper_band, 'g:', alpha=0.5, label='Upper BB')
    ax1.plot(range(len(df)), lower_band, 'r:', alpha=0.5, label='Lower BB')
    
    # Customize the main plot
    ax1.set_xlim(-1, len(df) + 1)
    ax1.set_ylim(df['low'].min() - y_margin, df['high'].max() + y_margin * 3)
    ax1.set_title(f'Candlestick Chart for {symbol}')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for patterns
    ax1.plot([], [], 'bo', label='Strong Doji', markersize=10)
    ax1.plot([], [], 'co', label='Weak Doji', markersize=10)
    ax1.plot([], [], 'g^', label='Bullish Hammer', markersize=10)
    ax1.plot([], [], 'r^', label='Bearish Hammer', markersize=10)
    ax1.plot([], [], 'gs', label='Bullish Engulfing', markersize=10)
    ax1.plot([], [], 'rs', label='Bearish Engulfing', markersize=10)
    ax1.plot([], [], 'g*', label='Morning Star', markersize=10)
    ax1.plot([], [], 'r*', label='Evening Star', markersize=10)
    ax1.plot([], [], 'gD', label='Three White Soldiers', markersize=10)
    ax1.plot([], [], 'rD', label='Three Black Crows', markersize=10)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # MACD subplot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(range(len(df)), macd, 'b-', label='MACD')
    ax2.plot(range(len(df)), signal, 'r--', label='Signal')
    ax2.bar(range(len(df)), histogram, color=['red' if h < 0 else 'green' for h in histogram], alpha=0.5)
    ax2.set_title('MACD')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # RSI subplot
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(range(len(df)), rsi, 'purple', label='RSI')
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax3.fill_between(range(len(df)), 70, rsi.where(rsi >= 70), color='red', alpha=0.3)
    ax3.fill_between(range(len(df)), 30, rsi.where(rsi <= 30), color='green', alpha=0.3)
    ax3.set_title('RSI')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Modified save functionality
    chart_dir = os.path.join("charts/technical_charts", symbol, datetime.now().strftime('%Y-%m'))
    os.makedirs(chart_dir, exist_ok=True)
    
    # Add metadata to filename
    filename = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}_RSI{df['RSI'].iloc[-1]:.1f}_MACD{df['MACD_HIST'].iloc[-1]:.2f}.png"
    chart_path = os.path.join(chart_dir, filename)
    
    # Add error handling and file rotation
    try:
        plt.savefig(chart_path, bbox_inches='tight', dpi=400)  # Reduced DPI for smaller files
        print(f"Chart saved to: {chart_path}")
        
        # Keep only last 5 charts per symbol
        all_charts = sorted([f for f in os.listdir(chart_dir) if f.startswith(symbol)], reverse=True)
        for old_chart in all_charts[5:]:
            os.remove(os.path.join(chart_dir, old_chart))
            
    except Exception as e:
        print(f"Error saving chart: {str(e)}")
        chart_path = None
        
    plt.close('all')  # Change this line from plt.close() to close all figures
    return chart_path

class StockAnalysisInterface:
    def __init__(self):
        """Initialize the interface with Alpaca credentials"""
        self.api_key = "your_api_key"
        self.secret_key = "your_secret_key"
        self.analyzer = AlpacaCandlestickAnalyzer(self.api_key, self.secret_key)
        self.fundamental_periods = 4  # Default fundamental periods
        
        # ANSI color codes for rich formatting
        self.colors = {
            'HEADER': '\033[95m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m',
            'END': '\033[0m'
        }

    def display_menu(self):
        """Display the main menu options"""
        menu = f"""
{self.colors['HEADER']}{self.colors['BOLD']}🚀 Stock Analysis Terminal 📊{self.colors['END']}
        
{self.colors['CYAN']}1.{self.colors['END']} Analyze Single Stock
{self.colors['CYAN']}2.{self.colors['END']} Compare Multiple Stocks
{self.colors['CYAN']}3.{self.colors['END']} View Saved Analysis
{self.colors['CYAN']}4.{self.colors['END']} Set Analysis Timeframe
{self.colors['RED']}5.{self.colors['END']} Exit

{self.colors['YELLOW']}Enter your choice (1-5):{self.colors['END']} """
        return input(menu)

    def get_stock_symbol(self):
        """Get stock symbol with validation"""
        while True:
            symbol = input(f"\n{self.colors['CYAN']}Enter stock symbol (e.g. AAPL):{self.colors['END']} ").strip().upper()
            if symbol and symbol.isalpha():
                return symbol
            print(f"{self.colors['RED']}Invalid symbol. Please try again.{self.colors['END']}")

    def analyze_single_stock(self):
        """Handle single stock analysis workflow"""
        symbol = self.get_stock_symbol()
        print(f"\n{self.colors['BOLD']}📊 Analyzing {symbol}...{self.colors['END']}\n")

        # Prompt for timeframe
        days_input = input(f"{self.colors['CYAN']}Enter number of days to analyze (press Enter for default 100):{self.colors['END']} ").strip()
        try:
            days = int(days_input) if days_input else 100
            if days < 1:
                print(f"{self.colors['RED']}Invalid number of days. Using default of 100 days.{self.colors['END']}")
                days = 100
        except ValueError:
            print(f"{self.colors['RED']}Invalid input. Using default of 100 days.{self.colors['END']}")
            days = 100

        # Get stock data using selected timeframe
        start_date = datetime.now() - timedelta(days=days)
        df = self.analyzer.get_stock_data(symbol, start_date)

        if df is None or len(df) < 5:
            print(f"{self.colors['RED']}❌ Insufficient data for analysis.{self.colors['END']}")
            return

        # Get fundamental analysis with selected periods
        print(f"\n{self.colors['YELLOW']}📊 Analyzing fundamentals...{self.colors['END']}")
        fundamental_analyzer = FundamentalAnalyzer(os.getenv("FINANCIAL_DATASETS_API_KEY"))
        fundamentals_df = fundamental_analyzer.get_fundamentals(symbol, limit=self.fundamental_periods)

        fundamental_metrics = {}
        if not fundamentals_df.empty:
            fundamental_chart_path = fundamental_analyzer.plot_fundamentals(fundamentals_df, symbol)
            print(f"{self.colors['GREEN']}✅ Fundamental analysis complete!{self.colors['END']}")
            
            # Save the fundamentals chart
            self.analyzer.storage.save_chart(symbol, fundamental_chart_path)
            
            # Extract latest fundamental metrics
            latest = fundamentals_df.iloc[-1]
            fundamental_metrics = {
                'revenue_growth': latest.get('revenue_growth_pct', 0),
                'gross_margin': latest.get('gross_margin', 0),
                'pe_ratio': latest.get('pe_ratio', 0),
                'eps_trend': "↑ Improving" if fundamentals_df['eps'].pct_change().iloc[-1] > 0 else "↓ Declining"
            }

        # Analyze patterns and market conditions
        df_with_patterns = self.analyzer.analyze_patterns(df)
        market_conditions = MarketAnalysis.analyze_market_conditions(df_with_patterns)
        market_summary = MarketAnalysis.generate_market_summary(market_conditions)

        # Display results with formatting
        print(f"\n{self.colors['HEADER']}🎯 MARKET ANALYSIS SUMMARY{self.colors['END']}")
        print(f"{self.colors['CYAN']}{market_summary}{self.colors['END']}")

        active_patterns = self.analyzer.get_active_patterns(df_with_patterns)
        if active_patterns:
            print(f"\n{self.colors['HEADER']}🔍 ACTIVE PATTERNS{self.colors['END']}")
            for pattern, signal in active_patterns.items():
                color = self.colors['GREEN'] if 'Bullish' in signal else self.colors['RED']
                print(f"{color}• {pattern}: {signal}{self.colors['END']}")

        # Generate and save chart
        print(f"\n{self.colors['YELLOW']}📈 Generating technical chart...{self.colors['END']}")
        chart_path = plot_candlestick(df_with_patterns, symbol)
        self.analyzer.storage.save_chart(symbol, chart_path)

        # Prepare technical metrics for AI analysis with proper key names
        latest_data = df_with_patterns.iloc[-1]
        tech_metrics = {
            'rsi': latest_data.get('RSI', 50),  # Default to 50 if missing
            'macd_hist': latest_data.get('MACD_HIST', 0),
            'bb_width': ((latest_data.get('BB_UPPER', 0) - latest_data.get('BB_LOWER', 0)) / 
                        latest_data.get('close', 1)) * 100,  # Prevent division by zero
            'close_price': latest_data.get('close', 0),
            'volume': latest_data.get('volume', 0),
            'active_patterns_count': len(active_patterns)
        }

        # Get AI analysis
        print(f"\n{self.colors['HEADER']}🤖 GENERATING AI INSIGHTS...{self.colors['END']}")
        
        # Get chart image analysis
        image_analysis = self.analyzer.analyst.analyze_chart_image(
            chart_path,
            market_summary,
            tech_metrics  # Use the properly formatted metrics
        )
        
        # Get comprehensive AI report with combined metrics
        ai_report = self.analyzer.analyst.generate_report(
            market_summary=market_summary,
            patterns=active_patterns,
            technicals={**tech_metrics, **fundamental_metrics}  # Combine both metric sets
        )

        # Display AI analysis with formatting
        print(f"\n{self.colors['HEADER']}🔮 AI CHART ANALYSIS{self.colors['END']}")
        print(f"{self.colors['CYAN']}{image_analysis}{self.colors['END']}")
        
        print(f"\n{self.colors['HEADER']}📝 AI MARKET REPORT{self.colors['END']}")
        print(f"{self.colors['CYAN']}{ai_report}{self.colors['END']}")
        
        input(f"\n{self.colors['BOLD']}Press Enter to continue...{self.colors['END']}")

    def compare_stocks(self):
        """Compare multiple stocks"""
        symbols = []
        while len(symbols) < 4:
            symbol = input(f"\n{self.colors['CYAN']}Enter stock symbol (or press Enter to finish):{self.colors['END']} ").strip().upper()
            if not symbol and len(symbols) >= 2:
                break
            if symbol and symbol.isalpha():
                symbols.append(symbol)

        print(f"\n{self.colors['BOLD']}📊 Comparing {', '.join(symbols)}...{self.colors['END']}\n")
        
        # Implement comparison logic here
        for symbol in symbols:
            self.analyze_single_stock()

    def view_saved_analysis(self):
        """View previously saved analysis"""
        symbol = self.get_stock_symbol()
        data, metadata = self.analyzer.storage.load_stock_data(symbol)
        
        if data is None:
            print(f"{self.colors['RED']}No saved analysis found for {symbol}{self.colors['END']}")
            return

        print(f"\n{self.colors['HEADER']}📋 SAVED ANALYSIS FOR {symbol}{self.colors['END']}")
        print(f"{self.colors['CYAN']}Last Updated: {metadata['last_updated']}{self.colors['END']}")
        
        # Display saved charts if available
        if 'charts' in metadata:
            print(f"\n{self.colors['YELLOW']}📈 Available Charts:{self.colors['END']}")
            for chart in metadata['charts']:
                print(f"• {chart['timestamp']}: {chart['path']}")

        input(f"\n{self.colors['BOLD']}Press Enter to continue...{self.colors['END']}")

    def set_analysis_timeframe(self):
        """Allow user to set custom analysis timeframe"""
        print(f"\n{self.colors['HEADER']}⏰ Set Analysis Timeframe{self.colors['END']}")
        
        # Display timeframe options
        print(f"""
{self.colors['CYAN']}1.{self.colors['END']} Last 7 Days
{self.colors['CYAN']}2.{self.colors['END']} Last 30 Days
{self.colors['CYAN']}3.{self.colors['END']} Last 90 Days
{self.colors['CYAN']}4.{self.colors['END']} Last 6 Months
{self.colors['CYAN']}5.{self.colors['END']} Last Year
{self.colors['CYAN']}6.{self.colors['END']} Custom Range
{self.colors['RED']}7.{self.colors['END']} Back to Main Menu
""")
        
        choice = input(f"{self.colors['YELLOW']}Enter your choice (1-7):{self.colors['END']} ")
        
        if choice == '1':
            self.timeframe = timedelta(days=7)
            print(f"{self.colors['GREEN']}✅ Timeframe set to last 7 days{self.colors['END']}")
        elif choice == '2':
            self.timeframe = timedelta(days=30)
            print(f"{self.colors['GREEN']}✅ Timeframe set to last 30 days{self.colors['END']}")
        elif choice == '3':
            self.timeframe = timedelta(days=90)
            print(f"{self.colors['GREEN']}✅ Timeframe set to last 90 days{self.colors['END']}")
        elif choice == '4':
            self.timeframe = timedelta(days=180)
            print(f"{self.colors['GREEN']}✅ Timeframe set to last 6 months{self.colors['END']}")
        elif choice == '5':
            self.timeframe = timedelta(days=365)
            print(f"{self.colors['GREEN']}✅ Timeframe set to last year{self.colors['END']}")
        elif choice == '6':
            self.set_custom_timeframe()
        elif choice == '7':
            return
        else:
            print(f"{self.colors['RED']}Invalid choice. Timeframe remains unchanged.{self.colors['END']}")
        
        # Set fundamental periods based on timeframe
        self.set_fundamental_periods()
        input(f"\n{self.colors['BOLD']}Press Enter to continue...{self.colors['END']}")

    def set_fundamental_periods(self):
        """Set number of periods for fundamental analysis based on timeframe"""
        if self.timeframe <= timedelta(days=30):
            self.fundamental_periods = 1  # Most recent quarter
        elif self.timeframe <= timedelta(days=90):
            self.fundamental_periods = 2  # Last 2 quarters
        elif self.timeframe <= timedelta(days=180):
            self.fundamental_periods = 4  # Last year
        else:
            self.fundamental_periods = 8  # Last 2 years
        
        # Allow user to override
        print(f"\n{self.colors['HEADER']}📊 Fundamental Analysis Periods{self.colors['END']}")
        print(f"Default periods based on timeframe: {self.fundamental_periods}")
        try:
            periods = int(input(f"{self.colors['CYAN']}Enter number of periods to analyze (1-8):{self.colors['END']} "))
            if 1 <= periods <= 8:
                self.fundamental_periods = periods
                print(f"{self.colors['GREEN']}✅ Fundamental periods set to {periods}{self.colors['END']}")
            else:
                print(f"{self.colors['RED']}Using default value of {self.fundamental_periods} periods{self.colors['END']}")
        except ValueError:
            print(f"{self.colors['RED']}Invalid input. Using default value of {self.fundamental_periods} periods{self.colors['END']}")

    def set_custom_timeframe(self):
        """Handle custom timeframe selection"""
        print(f"\n{self.colors['HEADER']}📅 Custom Timeframe{self.colors['END']}")
        
        while True:
            try:
                days = int(input(f"{self.colors['CYAN']}Enter number of days (1-365):{self.colors['END']} "))
                if 1 <= days <= 365:
                    self.timeframe = timedelta(days=days)
                    print(f"{self.colors['GREEN']}✅ Timeframe set to last {days} days{self.colors['END']}")
                    break
                else:
                    print(f"{self.colors['RED']}Please enter a number between 1 and 365{self.colors['END']}")
            except ValueError:
                print(f"{self.colors['RED']}Invalid input. Please enter a number.{self.colors['END']}")

    def run(self):
        """Main interface loop"""
        while True:
            choice = self.display_menu()
            
            if choice == '1':
                self.analyze_single_stock()
            elif choice == '2':
                self.compare_stocks()
            elif choice == '3':
                self.view_saved_analysis()
            elif choice == '4':
                self.set_analysis_timeframe()
            elif choice == '5':
                print(f"\n{self.colors['GREEN']}👋 Thank you for using Stock Analysis Terminal!{self.colors['END']}")
                break
            else:
                print(f"{self.colors['RED']}Invalid choice. Please try again.{self.colors['END']}")

def main():
    """Modified main function to use the new interface"""
    try:
        interface = StockAnalysisInterface()
        interface.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Details:", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
