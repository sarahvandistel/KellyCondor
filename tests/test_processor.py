"""
Unit tests for the Processor class.
"""

import pytest
import pandas as pd
import numpy as np
from kellycondor.processor import Processor


class TestProcessor:
    """Test cases for Processor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Processor()
    
    def test_init(self):
        """Test Processor initialization."""
        assert self.processor.data_cache == {}
    
    def test_calculate_iv_rank_insufficient_data(self):
        """Test IV rank calculation with insufficient data."""
        historical_iv = pd.Series([0.2, 0.3, 0.25])  # Less than 20 data points
        iv_rank = self.processor.calculate_iv_rank(historical_iv)
        assert iv_rank == 0.5
    
    def test_calculate_iv_rank_sufficient_data(self):
        """Test IV rank calculation with sufficient data."""
        # Create 25 days of IV data
        historical_iv = pd.Series(np.random.uniform(0.1, 0.5, 25))
        current_iv = 0.3
        historical_iv.iloc[-1] = current_iv
        
        iv_rank = self.processor.calculate_iv_rank(historical_iv)
        assert 0 <= iv_rank <= 1
    
    def test_calculate_skew(self):
        """Test volatility skew calculation."""
        call_iv = 0.25
        put_iv = 0.35
        skew = self.processor.calculate_skew(call_iv, put_iv)
        assert abs(skew - 0.1) < 1e-10
    
    def test_process_market_data_empty(self):
        """Test market data processing with empty data."""
        data = {}
        result = self.processor.process_market_data(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0 