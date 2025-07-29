"""
Databento Historical Data Integration for KellyCondor Backtesting

This module provides historical data retrieval from Databento for backtesting
the KellyCondor strategy with real market data.
"""

import os
import logging
import operator
import pathlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import databento as db

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
        
        self.client = db.Historical(self.api_key)
        logger.info("Initialized Databento historical data client")
    
    def get_symbol_mapping(self, symbols: List[str], dataset: str = "GLBX.MDP3") -> Dict[str, str]:
        """Get symbol mapping for given symbols."""
        try:
            # For now, return a simple mapping since symbology might not be available
            # In practice, you'd use the actual symbology API
            mapping = {symbol: symbol for symbol in symbols}
            logger.info(f"Retrieved symbol mapping for {len(symbols)} symbols")
            return mapping
        except Exception as e:
            logger.error(f"Failed to get symbol mapping: {e}")
            return {}
    
    def fetch_trade_data(self, config: HistoricalDataConfig) -> pd.DataFrame:
        """Fetch historical trade data using batch API."""
        try:
            logger.info(f"Fetching trade data for {config.symbol} from {config.start_date} to {config.end_date}")
            
            # Submit batch job
            new_job = self.client.batch.submit_job(
                dataset=config.dataset,
                start=config.start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                end=config.end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                symbols=config.symbol,
                schema="trades",
                split_duration="day",
                stype_in="parent",
            )
            
            # Get job ID
            new_job_id: str = new_job["id"]
            logger.info(f"Submitted batch job with ID: {new_job_id}")
            
            # Wait for job to complete
            logger.info("Waiting for batch job to complete...")
            while True:
                done_jobs = list(map(operator.itemgetter("id"), self.client.batch.list_jobs("done")))
                if new_job_id in done_jobs:
                    break
                time.sleep(1.0)
            
            logger.info("Batch job completed, downloading files...")
            
            # Download files
            downloaded_files = self.client.batch.download(
                job_id=new_job_id,
                output_dir=pathlib.Path.cwd(),
            )
            
            # Load data from files
            all_data = []
            for file in sorted(downloaded_files):
                if file.name.endswith(".dbn.zst"):
                    data = db.DBNStore.from_file(file)
                    df = data.to_df()
                    all_data.append(df)
                    logger.info(f"{file.name} contains {len(df):,d} records")
            
            if not all_data:
                logger.warning(f"No trade data found for {config.symbol}")
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Process and clean data
            df = self._process_trade_data(combined_df)
            logger.info(f"Retrieved {len(df)} total trade records for {config.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch trade data: {e}")
            return pd.DataFrame()
    
    def fetch_bar_data(self, config: HistoricalDataConfig) -> pd.DataFrame:
        """Fetch historical bar data (OHLCV) using batch API."""
        try:
            logger.info(f"Fetching bar data for {config.symbol} from {config.start_date} to {config.end_date}")
            
            if not config.interval:
                raise ValueError("Interval required for bar data")
            
            # Submit batch job for bars
            new_job = self.client.batch.submit_job(
                dataset=config.dataset,
                start=config.start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                end=config.end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                symbols=config.symbol,
                schema="ohlcv",
                split_duration="day",
                stype_in="parent",
            )
            
            # Get job ID
            new_job_id: str = new_job["id"]
            logger.info(f"Submitted batch job with ID: {new_job_id}")
            
            # Wait for job to complete
            logger.info("Waiting for batch job to complete...")
            while True:
                done_jobs = list(map(operator.itemgetter("id"), self.client.batch.list_jobs("done")))
                if new_job_id in done_jobs:
                    break
                time.sleep(1.0)
            
            logger.info("Batch job completed, downloading files...")
            
            # Download files
            downloaded_files = self.client.batch.download(
                job_id=new_job_id,
                output_dir=pathlib.Path.cwd(),
            )
            
            # Load data from files
            all_data = []
            for file in sorted(downloaded_files):
                if file.name.endswith(".dbn.zst"):
                    data = db.DBNStore.from_file(file)
                    df = data.to_df()
                    all_data.append(df)
                    logger.info(f"{file.name} contains {len(df):,d} records")
            
            if not all_data:
                logger.warning(f"No bar data found for {config.symbol}")
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Process and clean data
            df = self._process_bar_data(combined_df)
            logger.info(f"Retrieved {len(df)} total bar records for {config.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch bar data: {e}")
            return pd.DataFrame()
    
    def _process_trade_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean trade data from Databento."""
        if data.empty:
            return data
        
        # Clean and process data
        df = data.copy()
        
        # Convert timestamp (Databento uses nanoseconds)
        if 'ts_event' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
        elif 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ns')
        
        # Handle different column names for price and size
        price_col = None
        size_col = None
        
        for col in df.columns:
            if 'price' in col.lower():
                price_col = col
            elif 'size' in col.lower() or 'quantity' in col.lower():
                size_col = col
        
        if price_col:
            df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        if size_col:
            df['size'] = pd.to_numeric(df[size_col], errors='coerce')
        
        # Ensure we have required columns
        if 'price' not in df.columns or 'size' not in df.columns:
            logger.warning("Missing price or size columns in trade data")
            return pd.DataFrame()
        
        # Remove invalid data
        df = df.dropna(subset=['price', 'size', 'timestamp'])
        df = df[df['price'] > 0]
        df = df[df['size'] > 0]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _process_bar_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean bar data from Databento."""
        if data.empty:
            return data
        
        # Clean and process data
        df = data.copy()
        
        # Convert timestamp (Databento uses nanoseconds)
        if 'ts_event' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
        elif 'ts' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ns')
        
        # Handle different column names for OHLCV
        ohlcv_mapping = {
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'],
            'close': ['close', 'c'],
            'volume': ['volume', 'vol', 'v']
        }
        
        for target_col, possible_cols in ohlcv_mapping.items():
            found_col = None
            for col in df.columns:
                if col.lower() in possible_cols:
                    found_col = col
                    break
            
            if found_col:
                df[target_col] = pd.to_numeric(df[found_col], errors='coerce')
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in bar data: {missing_cols}")
            return pd.DataFrame()
        
        # Remove invalid data
        price_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=price_cols + ['volume', 'timestamp'])
        df = df[df[price_cols].gt(0).all(axis=1)]
        df = df[df['volume'] > 0]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_option_chain_data(self, underlying: str, expiry_date: datetime, 
                             dataset: str = "OPRA.PILLAR") -> pd.DataFrame:
        """Fetch option chain data for a specific expiry using batch API."""
        try:
            logger.info(f"Fetching option chain for {underlying} expiring {expiry_date}")
            
            # Format expiry date for option symbols
            expiry_str = expiry_date.strftime("%y%m%d")
            
            # Get all option symbols for the underlying
            symbols = self._get_option_symbols(underlying, expiry_str, dataset)
            if not symbols:
                logger.warning(f"No option symbols found for {underlying}")
                return pd.DataFrame()
            
            # Submit batch job for options
            new_job = self.client.batch.submit_job(
                dataset=dataset,
                start=(expiry_date - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S"),
                end=expiry_date.strftime("%Y-%m-%dT%H:%M:%S"),
                symbols=",".join(symbols),
                schema="trades",
                split_duration="day",
                stype_in="parent",
            )
            
            # Get job ID
            new_job_id: str = new_job["id"]
            logger.info(f"Submitted batch job with ID: {new_job_id}")
            
            # Wait for job to complete
            logger.info("Waiting for batch job to complete...")
            while True:
                done_jobs = list(map(operator.itemgetter("id"), self.client.batch.list_jobs("done")))
                if new_job_id in done_jobs:
                    break
                time.sleep(1.0)
            
            logger.info("Batch job completed, downloading files...")
            
            # Download files
            downloaded_files = self.client.batch.download(
                job_id=new_job_id,
                output_dir=pathlib.Path.cwd(),
            )
            
            # Load data from files
            all_data = []
            for file in sorted(downloaded_files):
                if file.name.endswith(".dbn.zst"):
                    data = db.DBNStore.from_file(file)
                    df = data.to_df()
                    all_data.append(df)
                    logger.info(f"{file.name} contains {len(df):,d} records")
            
            if not all_data:
                logger.warning(f"No option data found for {underlying}")
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Process option data
            df = self._process_option_data(combined_df, underlying)
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