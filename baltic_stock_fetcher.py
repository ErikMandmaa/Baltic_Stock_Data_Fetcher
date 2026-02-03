"""
Baltic Stock Data Fetcher - Realistic Implementation

Nasdaq Baltic doesn't provide a free public CSV download API.
This module provides working alternatives to get the data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Known ISINs for Baltic stocks
BALTIC_STOCKS = {
    'TKM1T': {'isin': 'EE3100004466', 'name': 'Tallinna Kaubamaja', 'country': 'EE'},
    'GRD1R': {'isin': 'LV0000100899', 'name': 'Grindeks', 'country': 'LV'},
    'APG1L': {'isin': 'LT0000128092', 'name': 'Apranga', 'country': 'LT'},
    'OLF1T': {'isin': 'EE3100004250', 'name': 'Olympic Entertainment', 'country': 'EE'},
    'PKG1T': {'isin': 'EE3100073657', 'name': 'Premia Foods', 'country': 'EE'},
}


class BalticDataFetcher:
    """
    Fetch Baltic stock data using available methods
    
    Reality check: Nasdaq Baltic doesn't have a free public API.
    Here are your actual options:
    
    1. Manual CSV download from nasdaqbaltic.com
    2. Use yfinance for cross-listed stocks (limited)
    3. Subscribe to Nasdaq Data Link (paid)
    4. Screen scraping (against ToS, not recommended)
    """
    
    def __init__(self):
        self.data_dir = './data'
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        import os
        os.makedirs(self.data_dir, exist_ok=True)
    
    def manual_download_instructions(self, ticker: str):
        """Print instructions for manual download"""
        if ticker not in BALTIC_STOCKS:
            print(f"Unknown ticker: {ticker}")
            print(f"Known tickers: {list(BALTIC_STOCKS.keys())}")
            return
        
        info = BALTIC_STOCKS[ticker]
        isin = info['isin']
        
        print(f"\n{'='*60}")
        print(f"Manual Download Instructions for {ticker}")
        print(f"{'='*60}")
        print(f"\n1. Go to: https://nasdaqbaltic.com/statistics/en/instrument/{isin}/trading")
        print(f"\n2. Look for the trading data table")
        print(f"\n3. Options:")
        print(f"   a) Copy table data and paste into Excel")
        print(f"   b) Use browser dev tools to export table")
        print(f"   c) Look for 'Export' or download icon on the page")
        print(f"\n4. Save as CSV with these columns:")
        print(f"   Date,Open,High,Low,Close,Volume")
        print(f"\n5. Save to: {self.data_dir}/{ticker}.csv")
        print(f"\n6. Then load with: fetcher.load_csv('{ticker}')")
        print(f"\n{'='*60}\n")
    
    def load_csv(self, ticker: str, filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        if filepath is None:
            filepath = f"{self.data_dir}/{ticker}.csv"
        
        try:
            df = pd.read_csv(filepath)
            
            # Normalize columns
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Map to standard names
            col_map = {
                'date': 'Date', 'close': 'Close', 'open': 'Open',
                'high': 'High', 'low': 'Low', 'volume': 'Volume'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            
            # Parse date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                df = df.sort_index()
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            self.manual_download_instructions(ticker)
            return None
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
    
    def try_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Try to fetch from Yahoo Finance
        Only works if stock is cross-listed
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(symbol)
            df = stock.history(period='2y')
            
            if not df.empty:
                logger.info(f"Found data on Yahoo Finance for {symbol}")
                return df
            
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
        except Exception as e:
            logger.debug(f"Yahoo Finance lookup failed: {e}")
        
        return None
    
    def get_sample_data(self, ticker: str, days: int = 500) -> pd.DataFrame:
        """
        Generate realistic sample data for testing
        NOT REAL DATA - for development only
        """
        logger.warning(f"Generating SAMPLE data for {ticker} - NOT REAL DATA")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # Start price based on typical Baltic stock prices
        base_prices = {'TKM1T': 8.5, 'GRD1R': 15.2, 'APG1L': 0.6}
        start_price = base_prices.get(ticker, 10.0)
        
        # Realistic parameters
        np.random.seed(hash(ticker) % 2**32)
        daily_vol = 0.015  # 1.5% daily volatility
        drift = 0.0001  # slight upward drift
        
        returns = np.random.normal(drift, daily_vol, days)
        prices = start_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
            'High': prices * (1 + np.random.uniform(0, 0.01, days)),
            'Low': prices * (1 - np.random.uniform(0, 0.01, days)),
            'Close': prices,
            'Volume': np.random.randint(5000, 50000, days)
        }, index=dates)
        
        return df


class StockAnalyzer:
    """Analyze stock data"""
    
    @staticmethod
    def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Get monthly summary stats"""
        monthly = df.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        monthly['return_pct'] = monthly['Close'].pct_change() * 100
        
        daily_returns = df['Close'].pct_change()
        monthly['volatility_pct'] = daily_returns.resample('M').std() * np.sqrt(252) * 100
        
        return monthly
    
    @staticmethod
    def current_metrics(df: pd.DataFrame) -> dict:
        """Calculate current metrics"""
        returns = df['Close'].pct_change()
        
        vol_30d = returns.tail(30).std() * np.sqrt(252) * 100
        vol_90d = returns.tail(90).std() * np.sqrt(252) * 100
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return {
            'current_price': df['Close'].iloc[-1],
            'vol_30d': vol_30d,
            'vol_90d': vol_90d,
            'max_drawdown': drawdown.min() * 100,
            'ytd_return': ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
        }


def main():
    print("Baltic Stock Data Fetcher")
    print("="*60)
    
    fetcher = BalticDataFetcher()
    analyzer = StockAnalyzer()
    
    ticker = 'TKM1T'
    
    print(f"\nAttempting to load {ticker}...")
    
    # Try to load from CSV
    df = fetcher.load_csv(ticker)
    
    # If that fails, try Yahoo Finance
    if df is None:
        print("\nTrying Yahoo Finance...")
        df = fetcher.try_yfinance(ticker)
    
    # If still nothing, offer sample data
    if df is None:
        print("\nNo real data available.")
        print("\nOptions:")
        print("1. Download manually (instructions shown above)")
        print("2. Use sample data for testing (y/n)?")
        
        choice = input("> ").lower()
        if choice == 'y':
            df = fetcher.get_sample_data(ticker)
    
    if df is not None:
        print(f"\n{'='*60}")
        print(f"Data Summary: {ticker}")
        print(f"{'='*60}")
        print(f"Records: {len(df)}")
        print(f"Period: {df.index.min().date()} to {df.index.max().date()}")
        
        print(f"\nLast 5 days:")
        print(df[['Close', 'Volume']].tail())
        
        metrics = analyzer.current_metrics(df)
        print(f"\nCurrent Metrics:")
        print(f"Price: €{metrics['current_price']:.2f}")
        print(f"30-day volatility: {metrics['vol_30d']:.1f}%")
        print(f"90-day volatility: {metrics['vol_90d']:.1f}%")
        print(f"Max drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"YTD return: {metrics['ytd_return']:.1f}%")
        
        monthly = analyzer.monthly_summary(df)
        print(f"\nMonthly Data (last 6 months):")
        print(monthly[['Close', 'return_pct', 'volatility_pct']].tail(6).round(2))
        
        # Save analysis
        monthly.to_csv('monthly_analysis.csv')
        print(f"\nSaved: monthly_analysis.csv")
    
    print(f"\n{'='*60}")
    print("Notes:")
    print("- Nasdaq Baltic has no free public API")
    print("- Manual download is currently required")
    print("- Alternative: Use paid Nasdaq Data Link subscription")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()