"""
Unit tests for the KellySizer class.
"""

import pytest
import numpy as np
from kellycondor.sizer import KellySizer


class TestKellySizer:
    """Test cases for KellySizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = KellySizer()
    
    def test_init(self):
        """Test KellySizer initialization."""
        assert self.sizer.max_kelly_fraction == 0.25
    
    def test_calculate_kelly_fraction_basic(self):
        """Test basic Kelly fraction calculation."""
        win_rate = 0.6
        avg_win = 100
        avg_loss = 50
        
        kelly_fraction = self.sizer.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        # The result should be capped at max_kelly_fraction (0.25)
        assert kelly_fraction == 0.25
    
    def test_calculate_kelly_fraction_zero_loss(self):
        """Test Kelly fraction calculation with zero loss."""
        kelly_fraction = self.sizer.calculate_kelly_fraction(0.5, 100, 0)
        assert kelly_fraction == 0.0
    
    def test_calculate_kelly_fraction_capped(self):
        """Test that Kelly fraction is capped at max_kelly_fraction."""
        # Use parameters that would give a very high Kelly fraction
        kelly_fraction = self.sizer.calculate_kelly_fraction(0.9, 1000, 10)
        assert kelly_fraction <= self.sizer.max_kelly_fraction
    
    def test_calculate_kelly_fraction_negative(self):
        """Test that negative Kelly fraction is capped at zero."""
        kelly_fraction = self.sizer.calculate_kelly_fraction(0.1, 10, 100)
        assert kelly_fraction == 0.0
    
    def test_size_position_basic(self):
        """Test basic position sizing."""
        iv_rank = 0.5
        skew = 0.1
        account_size = 100000
        
        result = self.sizer.size_position(iv_rank, skew, account_size)
        
        assert "kelly_fraction" in result
        assert "position_size" in result
        assert "max_risk_amount" in result
        assert "iv_rank" in result
        assert "skew" in result
        
        assert result["iv_rank"] == iv_rank
        assert result["skew"] == skew
        assert result["max_risk_amount"] == account_size * 0.02
    
    def test_size_position_high_iv_rank(self):
        """Test position sizing with high IV rank (more conservative)."""
        result_high_iv = self.sizer.size_position(0.9, 0.1, 100000)
        result_low_iv = self.sizer.size_position(0.1, 0.1, 100000)
        
        # Higher IV rank should result in lower Kelly fraction
        assert result_high_iv["kelly_fraction"] <= result_low_iv["kelly_fraction"]
    
    def test_size_position_high_skew(self):
        """Test position sizing with high skew (more conservative)."""
        result_high_skew = self.sizer.size_position(0.5, 0.5, 100000)
        result_low_skew = self.sizer.size_position(0.5, 0.0, 100000)
        
        # Higher skew should result in lower Kelly fraction
        assert result_high_skew["kelly_fraction"] <= result_low_skew["kelly_fraction"] 