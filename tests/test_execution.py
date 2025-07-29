"""
Unit tests for the execution module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from kellycondor.execution import (
    IBKRClient,
    LiveIVSkewProcessor,
    LiveKellySizer,
    IBKRTradeExecutor,
    IronCondorOrder,
    DatabentoStream
)


class TestIronCondorOrder:
    """Test cases for IronCondorOrder dataclass."""
    
    def test_init(self):
        """Test IronCondorOrder initialization."""
        order = IronCondorOrder(
            symbol="SPX",
            expiry="20241220",
            call_strike=4550,
            put_strike=4450,
            call_spread=25,
            put_spread=25,
            quantity=1
        )
        
        assert order.symbol == "SPX"
        assert order.expiry == "20241220"
        assert order.call_strike == 4550
        assert order.put_strike == 4450
        assert order.call_spread == 25
        assert order.put_spread == 25
        assert order.quantity == 1
        assert order.status == "PENDING"
        assert order.order_id is None
        assert order.fill_price is None


class TestLiveIVSkewProcessor:
    """Test cases for LiveIVSkewProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strikes = [4400, 4450, 4500, 4550, 4600]
        self.expiries = ["20241220"]
        self.processor = LiveIVSkewProcessor(self.strikes, self.expiries)
    
    def test_init(self):
        """Test LiveIVSkewProcessor initialization."""
        assert self.processor.strikes == self.strikes
        assert self.processor.expiries == self.expiries
        assert self.processor.current_iv_data == {}
        assert self.processor.current_skew == 0.0
        assert self.processor.current_iv_rank == 0.5
    
    def test_process_tick_call(self):
        """Test processing call option tick."""
        tick = {
            "symbol": "SPX",
            "strike": 4500,
            "expiry": "20241220",
            "iv": 0.25,
            "type": "C"
        }
        
        self.processor.process_tick(tick)
        
        key = "SPX_4500_20241220_C"
        assert key in self.processor.current_iv_data
        assert self.processor.current_iv_data[key] == 0.25
    
    def test_process_tick_put(self):
        """Test processing put option tick."""
        tick = {
            "symbol": "SPX",
            "strike": 4500,
            "expiry": "20241220",
            "iv": 0.30,
            "type": "P"
        }
        
        self.processor.process_tick(tick)
        
        key = "SPX_4500_20241220_P"
        assert key in self.processor.current_iv_data
        assert self.processor.current_iv_data[key] == 0.30
    
    def test_calculate_skew(self):
        """Test skew calculation from call and put IV."""
        # Add call IV
        call_tick = {
            "symbol": "SPX",
            "strike": 4500,
            "expiry": "20241220",
            "iv": 0.25,
            "type": "C"
        }
        self.processor.process_tick(call_tick)
        
        # Add put IV
        put_tick = {
            "symbol": "SPX",
            "strike": 4500,
            "expiry": "20241220",
            "iv": 0.30,
            "type": "P"
        }
        self.processor.process_tick(put_tick)
        
        # Check skew calculation
        assert abs(self.processor.current_skew - 0.05) < 1e-10
    
    def test_estimate_iv_rank_low(self):
        """Test IV rank estimation for low IV."""
        rank = self.processor._estimate_iv_rank(0.10)
        assert rank == 0.2
    
    def test_estimate_iv_rank_medium(self):
        """Test IV rank estimation for medium IV."""
        rank = self.processor._estimate_iv_rank(0.20)
        assert rank == 0.5
    
    def test_estimate_iv_rank_high(self):
        """Test IV rank estimation for high IV."""
        rank = self.processor._estimate_iv_rank(0.30)
        assert rank == 0.8


class TestLiveKellySizer:
    """Test cases for LiveKellySizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strikes = [4400, 4450, 4500, 4550, 4600]
        self.expiries = ["20241220"]
        self.processor = LiveIVSkewProcessor(self.strikes, self.expiries)
        
        self.win_rate_table = {
            "low_iv": 0.65,
            "medium_iv": 0.60,
            "high_iv": 0.55
        }
        self.sizer = LiveKellySizer(self.processor, self.win_rate_table)
    
    def test_init(self):
        """Test LiveKellySizer initialization."""
        assert self.sizer.processor == self.processor
        assert self.sizer.win_rate_table == self.win_rate_table
        assert self.sizer.account_size == 100000
    
    def test_get_iv_bucket_low(self):
        """Test IV bucket classification for low IV."""
        bucket = self.sizer._get_iv_bucket(0.2)
        assert bucket == "low_iv"
    
    def test_get_iv_bucket_medium(self):
        """Test IV bucket classification for medium IV."""
        bucket = self.sizer._get_iv_bucket(0.5)
        assert bucket == "medium_iv"
    
    def test_get_iv_bucket_high(self):
        """Test IV bucket classification for high IV."""
        bucket = self.sizer._get_iv_bucket(0.8)
        assert bucket == "high_iv"
    
    def test_get_current_sizing(self):
        """Test current sizing calculation."""
        # Set up processor state
        self.processor.current_iv_rank = 0.6
        self.processor.current_skew = 0.05
        
        sizing = self.sizer.get_current_sizing()
        
        assert "kelly_fraction" in sizing
        assert "position_size" in sizing
        assert "max_risk_amount" in sizing
        assert sizing["iv_rank"] == 0.6
        assert sizing["skew"] == 0.05


class TestDatabentoStream:
    """Test cases for DatabentoStream."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stream = DatabentoStream("test_key")
    
    def test_init(self):
        """Test DatabentoStream initialization."""
        assert self.stream.api_key == "test_key"
        assert not self.stream.running
    
    def test_stop(self):
        """Test stopping the data stream."""
        self.stream.running = True
        self.stream.stop()
        assert not self.stream.running
    
    @patch('time.sleep')
    def test_stream_iv_and_skew(self, mock_sleep):
        """Test the data stream generator."""
        mock_sleep.return_value = None  # Don't actually sleep
        
        # Get first few ticks
        ticks = []
        for i, tick in enumerate(self.stream.stream_iv_and_skew()):
            if i >= 3:  # Get 3 ticks
                break
            ticks.append(tick)
        
        assert len(ticks) == 3
        
        for tick in ticks:
            assert "symbol" in tick
            assert "strike" in tick
            assert "expiry" in tick
            assert "iv" in tick
            assert "type" in tick
            assert "timestamp" in tick
            assert tick["symbol"] == "SPX"
            assert tick["strike"] == 4500
            assert tick["expiry"] == "20241220"
            assert tick["type"] in ["C", "P"]


class TestIBKRClient:
    """Test cases for IBKRClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = IBKRClient()
    
    def test_init(self):
        """Test IBKRClient initialization."""
        assert not self.client.connected
        assert self.client.next_order_id is None
        assert self.client.orders == {}
        assert self.client.positions == {}
        assert self.client.account_info == {}
    
    def test_next_valid_id(self):
        """Test nextValidId callback."""
        self.client.nextValidId(123)
        assert self.client.next_order_id == 123
    
    def test_order_status(self):
        """Test orderStatus callback."""
        # Create a mock order
        order = IronCondorOrder(
            symbol="SPX",
            expiry="20241220",
            call_strike=4550,
            put_strike=4450,
            call_spread=25,
            put_spread=25,
            quantity=1,
            order_id=123
        )
        self.client.orders[123] = order
        
        # Simulate order status update
        self.client.orderStatus(
            orderId=123,
            status="Filled",
            filled=1.0,
            remaining=0.0,
            avgFillPrice=2.50,
            permId=456,
            parentId=0,
            lastFillPrice=2.50,
            clientId=1,
            whyHeld="",
            mktCapPrice=0.0
        )
        
        assert order.status == "Filled"
        assert order.fill_price == 2.50
        assert order.timestamp is not None
    
    def test_account_summary(self):
        """Test accountSummary callback."""
        self.client.accountSummary(
            reqId=1,
            account="DU123456",
            tag="NetLiquidation",
            value="100000.00",
            currency="USD"
        )
        
        assert "NetLiquidation" in self.client.account_info
        assert self.client.account_info["NetLiquidation"]["value"] == "100000.00"
        assert self.client.account_info["NetLiquidation"]["currency"] == "USD"


class TestIBKRTradeExecutor:
    """Test cases for IBKRTradeExecutor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = Mock()
        self.client.connected = True
        self.client.next_order_id = 100
        
        self.strikes = [4400, 4450, 4500, 4550, 4600]
        self.expiries = ["20241220"]
        self.processor = LiveIVSkewProcessor(self.strikes, self.expiries)
        
        self.win_rate_table = {
            "low_iv": 0.65,
            "medium_iv": 0.60,
            "high_iv": 0.55
        }
        self.sizer = LiveKellySizer(self.processor, self.win_rate_table)
        
        self.executor = IBKRTradeExecutor(self.client, self.processor, self.sizer)
    
    def test_init(self):
        """Test IBKRTradeExecutor initialization."""
        assert self.executor.client == self.client
        assert self.executor.processor == self.processor
        assert self.executor.sizer == self.sizer
        assert self.executor.active_orders == {}
        assert self.executor.trade_history == []
    
    def test_create_iron_condor_order(self):
        """Test iron condor order creation."""
        sizing = {
            "position_size": 1000,
            "max_risk_amount": 2000
        }
        
        order = self.executor._create_iron_condor_order("SPX", "20241220", sizing)
        
        assert order.symbol == "SPX"
        assert order.expiry == "20241220"
        assert order.call_strike == 4550  # current_price + 50
        assert order.put_strike == 4450   # current_price - 50
        assert order.call_spread == 25
        assert order.put_spread == 25
        assert order.quantity == 0  # Simplified calculation
    
    def test_submit_iron_condor_not_connected(self):
        """Test submitting order when not connected."""
        self.client.connected = False
        
        result = self.executor.submit_iron_condor()
        assert result is None
    
    @patch.object(IBKRTradeExecutor, '_submit_order')
    def test_submit_iron_condor_success(self, mock_submit):
        """Test successful iron condor submission."""
        mock_submit.return_value = True
        
        # Set up processor state
        self.processor.current_iv_rank = 0.6
        self.processor.current_skew = 0.05
        
        result = self.executor.submit_iron_condor()
        
        assert result is not None
        # The order_id will be set by the mock, so we just check the order exists
        assert result in self.executor.active_orders.values() 