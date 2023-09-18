import yfinance
import numpy as np
import pandas as pd

def download_yfinance(ticker:str, start_date: str, end_date:str) -> pd.DataFrame:
    """Extract histtorical prices from yfinance"""
    data = yf.download(ticker_symbol, start='2023-01-01', end='2023-08-01')

    return data

def compute_annualised_volatility(data: np.array, rolling_window: int=21, trading_days: int = 252) -> np.array:
    if rolling_window > len(data):
        raise ValueError("Rolling window larger than dataset.")
    
    # Compute daily returns
    daily_returns = np.diff(data) / data[:-1]
    
    # Calculate rolling standard deviation of daily returns
    volatilities = np.array([np.std(daily_returns[max(0, i-trading_days):i]) for i in range(1, len(daily_returns) + 1)])
    
    # Annualize the volatility
    annualized_volatilities = volatilities * np.sqrt(trading_days)
    
    return annualized_volatilities[1:]
    