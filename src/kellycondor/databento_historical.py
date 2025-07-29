"""
Databento Historical Data Integration for KellyCondor Backtesting

This module provides historical data retrieval from Databento for backtesting
the KellyCondor strategy with real market data.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from databento import Historical, Symbology

logger = logging.getLogger(__name__)

@dataclass
class HistoricalDataConfig:
    """Configuration for historical data retrieval."""
    symbol: str
    start_date: datetime
    end_date: datetime
    dataset: str = "GLBX.MDP3"  # CME Globex
    stype_in: str = "trade"  # Trade data
    stype_out: str = "symbol"  # Symbol mapping
    interval: Optional[str] = None  # For bar data
    price_type: str = "trade"  # Trade price
    max_records: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        
        if self.interval and self.interval not in ["1s", "1m", "1h", "1d"]:
            raise ValueError("interval must be one of: 1s, 1m, 1h, 1d")

@dataclass
class MarketDataPoint:
    """Represents a single market data point."""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    side: str  # 'buy' or 'sell'
    exchange: str
    sequence: int
    
    @classmethod
    def from_databento_record(cls, record: Dict[str, Any]) -> 'MarketDataPoint':
        """Create from Databento record."""
        return cls(
            timestamp=record.get('ts_event', record.get('ts', datetime.now())),
            symbol=record.get('symbol', ''),
            price=float(record.get('price', 0)),
            volume=int(record.get('size', 0)),
            side=record.get('side', 'unknown'),
            exchange=record.get('exchange', ''),
            sequence=int(record.get('sequence', 0))
        )

class DatabentoHistoricalData:
    """Handles historical data retrieval from Databento."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key."""
        self.api_key = api_key or os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("Databento API key required. Set DATABENTO_API_KEY environment variable.")
        
        self.client = Historical(self.api_key)
        logger.info("Initialized Databento historical data client")
    
    def get_symbol_mapping(self, symbols: List[str], dataset: str = "GLBX.MDP3") -> Dict[str, str]:
        """Get symbol mapping for given symbols."""
        try:
            symbology = Symbology(
                dataset=dataset,
                symbols=symbols,
                stype_in="symbol",
                stype_out="symbol",
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            mapping = self.client.symbology(symbology)
            logger.info(f"Retrieved symbol mapping for {len(symbols)} symbols")
            return mapping
        except Exception as e:
            logger.error(f"Failed to get symbol mapping: {e}")
            return {}
    
    def fetch_trade_data(self, config: HistoricalDataConfig) -> pd.DataFrame:
        """Fetch historical trade data."""
        try:
            logger.info(f"Fetching trade data for {config.symbol} from {config.start_date} to {config.end_date}")
            
            # Get symbol mapping
            mapping = self.get_symbol_mapping([config.symbol], config.dataset)
            if not mapping:
                logger.warning(f"No symbol mapping found for {config.symbol}")
                return pd.DataFrame()
            
            # Fetch trade data
            data = self.client.timeseries.get_range(
                dataset=config.dataset,
                symbols=[config.symbol],
                stype_in=config.stype_in,
                stype_out=config.stype_out,
                start=config.start_date,
                end=config.end_date,
                max_records=config.max_records
            )
            
            if data.empty:
                logger.warning(f"No trade data found for {config.symbol}")
                return pd.DataFrame()
            
            # Process and clean data
            df = self._process_trade_data(data)
            logger.info(f"Retrieved {len(df)} trade records for {config.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch trade data: {e}")
            return pd.DataFrame()
    
    def fetch_bar_data(self, config: HistoricalDataConfig) -> pd.DataFrame:
        """Fetch historical bar data (OHLCV)."""
        try:
            logger.info(f"Fetching bar data for {config.symbol} from {config.start_date} to {config.end_date}")
            
            if not config.interval:
                raise ValueError("Interval required for bar data")
            
            # Get symbol mapping
            mapping = self.get_symbol_mapping([config.symbol], config.dataset)
            if not mapping:
                logger.warning(f"No symbol mapping found for {config.symbol}")
                return pd.DataFrame()
            
            # Fetch bar data
            data = self.client.timeseries.get_range(
                dataset=config.dataset,
                symbols=[config.symbol],
                stype_in=config.stype_in,
                stype_out=config.stype_out,
                start=config.start_date,
                end=config.end_date,
                interval=config.interval,
                price_type=config.price_type,
                max_records=config.max_records
            )
            
            if data.empty:
                logger.warning(f"No bar data found for {config.symbol}")
                return pd.DataFrame()
            
            # Process and clean data
            df = self._process_bar_data(data)
            logger.info(f"Retrieved {len(df)} bar records for {config.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch bar data: {e}")
            return pd.DataFrame()
    
    def _process_trade_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean trade data."""
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_cols = ['ts_event', 'symbol', 'price', 'size']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns in trade data: {missing_cols}")
            return pd.DataFrame()
        
        # Clean and process data
        df = data.copy()
        
        # Convert timestamp
        if 'ts_event' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
        elif 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ns')
        
        # Ensure numeric types
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['price', 'size', 'timestamp'])
        df = df[df['price'] > 0]
        df = df[df['size'] > 0]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _process_bar_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean bar data."""
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_cols = ['ts_event', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns in bar data: {missing_cols}")
            return pd.DataFrame()
        
        # Clean and process data
        df = data.copy()
        
        # Convert timestamp
        if 'ts_event' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
        elif 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ns')
        
        # Ensure numeric types
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=price_cols + ['volume', 'timestamp'])
        df = df[df[price_cols].gt(0).all(axis=1)]
        df = df[df['volume'] > 0]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_option_chain_data(self, underlying: str, expiry_date: datetime, 
                             dataset: str = "OPRA.PILLAR") -> pd.DataFrame:
        """Fetch option chain data for a specific expiry."""
        try:
            logger.info(f"Fetching option chain for {underlying} expiring {expiry_date}")
            
            # Format expiry date for option symbols
            expiry_str = expiry_date.strftime("%y%m%d")
            
            # Get all option symbols for the underlying
            symbols = self._get_option_symbols(underlying, expiry_str, dataset)
            if not symbols:
                logger.warning(f"No option symbols found for {underlying}")
                return pd.DataFrame()
            
            # Fetch option data
            data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=symbols,
                stype_in="symbol",
                stype_out="symbol",
                start=expiry_date - timedelta(days=30),
                end=expiry_date,
                max_records=10000
            )
            
            if data.empty:
                logger.warning(f"No option data found for {underlying}")
                return pd.DataFrame()
            
            # Process option data
            df = self._process_option_data(data, underlying)
            logger.info(f"Retrieved {len(df)} option records for {underlying}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch option chain data: {e}")
            return pd.DataFrame()
    
    def _get_option_symbols(self, underlying: str, expiry: str, dataset: str) -> List[str]:
        """Get option symbols for underlying and expiry."""
        # This is a simplified implementation
        # In practice, you'd need to query available symbols
        strikes = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
        symbols = []
        
        for strike in strikes:
            # Call options
            call_symbol = f"{underlying}{expiry}C{strike:05d}"
            symbols.append(call_symbol)
            
            # Put options
            put_symbol = f"{underlying}{expiry}P{strike:05d}"
            symbols.append(put_symbol)
        
        return symbols
    
    def _process_option_data(self, data: pd.DataFrame, underlying: str) -> pd.DataFrame:
        """Process option data and extract Greeks."""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Basic processing
        df = self._process_trade_data(df)
        
        # Add option-specific fields
        df['underlying'] = underlying
        df['option_type'] = df['symbol'].str[-1].map({'C': 'call', 'P': 'put'})
        
        # Extract strike price from symbol (simplified)
        df['strike'] = df['symbol'].str[-6:-1].astype(float) / 1000
        
        # Calculate implied volatility (simplified)
        df['iv'] = self._calculate_implied_volatility(df)
        
        return df
    
    def _calculate_implied_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate implied volatility (simplified implementation)."""
        # This is a placeholder - in practice you'd use a proper IV calculation
        # For now, return a random IV between 0.1 and 0.5
        return np.random.uniform(0.1, 0.5, len(df))

def create_historical_data_client(api_key: Optional[str] = None) -> DatabentoHistoricalData:
    """Factory function to create historical data client."""
    return DatabentoHistoricalData(api_key)

def get_backtest_data(symbol: str, start_date: datetime, end_date: datetime,
                     api_key: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to get backtest data."""
    client = create_historical_data_client(api_key)
    
    config = HistoricalDataConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        dataset="GLBX.MDP3",
        stype_in="trade",
        stype_out="symbol"
    )
    
    return client.fetch_trade_data(config) 