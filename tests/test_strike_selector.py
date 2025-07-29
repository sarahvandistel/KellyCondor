"""
Tests for advanced strike selector functionality.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from kellycondor.strike_selector import (
    AdvancedStrikeSelector,
    RotatingStrikeSelector,
    StrikeSelectionConfig,
    StrikeSelectionResult,
    SkewBucket,
    IVPercentile,
    create_default_strike_selector,
    create_rotating_strike_selector,
    create_custom_strike_selector
)


class TestStrikeSelectionConfig:
    """Test StrikeSelectionConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = StrikeSelectionConfig()
        
        assert config.iv_low_threshold == 0.3
        assert config.iv_high_threshold == 0.7
        assert config.skew_low_threshold == 0.05
        assert config.skew_high_threshold == 0.15
        assert config.min_wing_distance == 10
        assert config.max_wing_distance == 100
        
        # Check default wing distances
        assert "low_iv" in config.wing_distances
        assert "medium_iv" in config.wing_distances
        assert "high_iv" in config.wing_distances
        
        # Check default spread widths
        assert "low_iv" in config.spread_widths
        assert "medium_iv" in config.spread_widths
        assert "high_iv" in config.spread_widths
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        custom_wing_distances = {
            "low_iv": {"low_skew": 10, "medium_skew": 15, "high_skew": 20},
            "medium_iv": {"low_skew": 15, "medium_skew": 25, "high_skew": 35},
            "high_iv": {"low_skew": 20, "medium_skew": 30, "high_skew": 45}
        }
        
        custom_spread_widths = {
            "low_iv": 10,
            "medium_iv": 15,
            "high_iv": 20
        }
        
        config = StrikeSelectionConfig(
            wing_distances=custom_wing_distances,
            spread_widths=custom_spread_widths,
            iv_low_threshold=0.25,
            iv_high_threshold=0.75
        )
        
        assert config.wing_distances == custom_wing_distances
        assert config.spread_widths == custom_spread_widths
        assert config.iv_low_threshold == 0.25
        assert config.iv_high_threshold == 0.75


class TestSkewBucket:
    """Test SkewBucket enum."""
    
    def test_skew_bucket_values(self):
        """Test skew bucket enum values."""
        assert SkewBucket.LOW_SKEW.value == "low_skew"
        assert SkewBucket.MEDIUM_SKEW.value == "medium_skew"
        assert SkewBucket.HIGH_SKEW.value == "high_skew"


class TestIVPercentile:
    """Test IVPercentile enum."""
    
    def test_iv_percentile_values(self):
        """Test IV percentile enum values."""
        assert IVPercentile.LOW_IV.value == "low_iv"
        assert IVPercentile.MEDIUM_IV.value == "medium_iv"
        assert IVPercentile.HIGH_IV.value == "high_iv"


class TestAdvancedStrikeSelector:
    """Test AdvancedStrikeSelector class."""
    
    def test_selector_creation(self):
        """Test creating a strike selector."""
        selector = AdvancedStrikeSelector()
        
        assert selector.config is not None
        assert selector.historical_performance == {}
        assert selector.selection_history == []
    
    def test_get_iv_percentile(self):
        """Test IV percentile determination."""
        selector = AdvancedStrikeSelector()
        
        # Test with historical data
        historical_ivs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Low IV
        result = selector.get_iv_percentile(0.2, historical_ivs)
        assert result == IVPercentile.LOW_IV
        
        # Medium IV
        result = selector.get_iv_percentile(0.5, historical_ivs)
        assert result == IVPercentile.MEDIUM_IV
        
        # High IV
        result = selector.get_iv_percentile(0.8, historical_ivs)
        assert result == IVPercentile.HIGH_IV
    
    def test_get_skew_bucket(self):
        """Test skew bucket determination."""
        selector = AdvancedStrikeSelector()
        
        # Low skew
        result = selector.get_skew_bucket(0.02)
        assert result == SkewBucket.LOW_SKEW
        
        # Medium skew
        result = selector.get_skew_bucket(0.1)
        assert result == SkewBucket.MEDIUM_SKEW
        
        # High skew
        result = selector.get_skew_bucket(0.2)
        assert result == SkewBucket.HIGH_SKEW
        
        # Negative skew (should use absolute value)
        result = selector.get_skew_bucket(-0.2)
        assert result == SkewBucket.HIGH_SKEW
    
    def test_get_wing_distance(self):
        """Test wing distance calculation."""
        selector = AdvancedStrikeSelector()
        
        # Test default configuration
        wing_distance = selector.get_wing_distance(IVPercentile.MEDIUM_IV, SkewBucket.MEDIUM_SKEW)
        assert wing_distance == 30  # Default medium_iv/medium_skew
        
        # Test custom configuration
        custom_config = StrikeSelectionConfig()
        custom_config.wing_distances["medium_iv"]["medium_skew"] = 35
        selector = AdvancedStrikeSelector(custom_config)
        
        wing_distance = selector.get_wing_distance(IVPercentile.MEDIUM_IV, SkewBucket.MEDIUM_SKEW)
        assert wing_distance == 35
    
    def test_get_spread_width(self):
        """Test spread width calculation."""
        selector = AdvancedStrikeSelector()
        
        spread_width = selector.get_spread_width(IVPercentile.MEDIUM_IV)
        assert spread_width == 20  # Default medium_iv spread width
    
    def test_calculate_strikes(self):
        """Test strike calculation."""
        selector = AdvancedStrikeSelector()
        current_price = 4500
        
        result = selector.calculate_strikes(
            current_price, 
            IVPercentile.MEDIUM_IV, 
            SkewBucket.MEDIUM_SKEW
        )
        
        assert isinstance(result, StrikeSelectionResult)
        assert result.call_strike == current_price + 30  # wing_distance
        assert result.put_strike == current_price - 30
        assert result.call_spread == 20  # spread_width
        assert result.put_spread == 20
        assert result.iv_percentile == IVPercentile.MEDIUM_IV
        assert result.skew_bucket == SkewBucket.MEDIUM_SKEW
        assert result.wing_distance == 30
        assert result.selection_score >= 0.0
        assert result.reasoning is not None
    
    def test_update_performance(self):
        """Test performance update."""
        selector = AdvancedStrikeSelector()
        
        # Create a selection result
        result = StrikeSelectionResult(
            call_strike=4530,
            put_strike=4470,
            call_spread=20,
            put_spread=20,
            iv_percentile=IVPercentile.MEDIUM_IV,
            skew_bucket=SkewBucket.MEDIUM_SKEW,
            wing_distance=30,
            selection_score=0.7,
            reasoning="Test reasoning"
        )
        
        # Update performance
        selector.update_performance(result, 100.0, True, 1000.0)
        
        key = f"{result.iv_percentile.value}_{result.skew_bucket.value}"
        performance = selector.historical_performance[key]
        
        assert performance["trades"] == 1
        assert performance["wins"] == 1
        assert performance["total_pnl"] == 100.0
        assert performance["win_rate"] == 1.0
        assert performance["avg_pnl"] == 100.0
    
    def test_get_top_ranked_combinations(self):
        """Test getting top-ranked combinations."""
        selector = AdvancedStrikeSelector()
        
        # Add some performance data
        result1 = StrikeSelectionResult(
            call_strike=4530, put_strike=4470, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.MEDIUM_IV, skew_bucket=SkewBucket.MEDIUM_SKEW,
            wing_distance=30, selection_score=0.7, reasoning="Test 1"
        )
        
        result2 = StrikeSelectionResult(
            call_strike=4540, put_strike=4460, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.HIGH_IV, skew_bucket=SkewBucket.LOW_SKEW,
            wing_distance=40, selection_score=0.8, reasoning="Test 2"
        )
        
        # Update performance
        selector.update_performance(result1, 50.0, True, 1000.0)
        selector.update_performance(result1, 30.0, True, 1000.0)
        selector.update_performance(result1, -20.0, False, 1000.0)
        
        selector.update_performance(result2, 100.0, True, 1000.0)
        selector.update_performance(result2, 80.0, True, 1000.0)
        selector.update_performance(result2, 60.0, True, 1000.0)
        
        # Get top combinations
        top_combinations = selector.get_top_ranked_combinations()
        
        assert len(top_combinations) == 2
        
        # Second combination should be ranked higher (better performance)
        assert top_combinations[0][0] == f"{IVPercentile.HIGH_IV.value}_{SkewBucket.LOW_SKEW.value}"
        assert top_combinations[1][0] == f"{IVPercentile.MEDIUM_IV.value}_{SkewBucket.MEDIUM_SKEW.value}"
    
    def test_select_optimal_strikes(self):
        """Test optimal strike selection."""
        selector = AdvancedStrikeSelector()
        current_price = 4500
        current_iv = 0.5
        current_skew = 0.1
        
        result = selector.select_optimal_strikes(current_price, current_iv, current_skew)
        
        assert isinstance(result, StrikeSelectionResult)
        assert result.call_strike > current_price
        assert result.put_strike < current_price
        assert result.iv_percentile in [IVPercentile.LOW_IV, IVPercentile.MEDIUM_IV, IVPercentile.HIGH_IV]
        assert result.skew_bucket in [SkewBucket.LOW_SKEW, SkewBucket.MEDIUM_SKEW, SkewBucket.HIGH_SKEW]
    
    def test_select_optimal_strikes_force_top_ranked(self):
        """Test optimal strike selection with top-ranked forcing."""
        selector = AdvancedStrikeSelector()
        
        # Add some performance data first
        result1 = StrikeSelectionResult(
            call_strike=4530, put_strike=4470, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.MEDIUM_IV, skew_bucket=SkewBucket.MEDIUM_SKEW,
            wing_distance=30, selection_score=0.7, reasoning="Test 1"
        )
        
        result2 = StrikeSelectionResult(
            call_strike=4540, put_strike=4460, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.HIGH_IV, skew_bucket=SkewBucket.LOW_SKEW,
            wing_distance=40, selection_score=0.8, reasoning="Test 2"
        )
        
        # Update performance to create ranking
        selector.update_performance(result1, 50.0, True, 1000.0)
        selector.update_performance(result2, 100.0, True, 1000.0)
        
        # Test with force_top_ranked=True
        current_price = 4500
        current_iv = 0.3  # Would normally be LOW_IV
        current_skew = 0.02  # Would normally be LOW_SKEW
        
        result = selector.select_optimal_strikes(
            current_price, current_iv, current_skew, force_top_ranked=True
        )
        
        # Should select the top-ranked combination (HIGH_IV, LOW_SKEW)
        assert result.iv_percentile == IVPercentile.HIGH_IV
        assert result.skew_bucket == SkewBucket.LOW_SKEW


class TestRotatingStrikeSelector:
    """Test RotatingStrikeSelector class."""
    
    def test_rotating_selector_creation(self):
        """Test creating a rotating strike selector."""
        selector = RotatingStrikeSelector(rotation_period=3)
        
        assert selector.rotation_period == 3
        assert selector.current_rotation_index == 0
        assert selector.rotation_history == []
    
    def test_select_rotating_strikes_no_data(self):
        """Test rotating selection with no historical data."""
        selector = RotatingStrikeSelector()
        current_price = 4500
        current_iv = 0.5
        current_skew = 0.1
        
        result = selector.select_rotating_strikes(current_price, current_iv, current_skew)
        
        # Should fall back to standard selection
        assert isinstance(result, StrikeSelectionResult)
        assert result.call_strike > current_price
        assert result.put_strike < current_price
    
    def test_select_rotating_strikes_with_data(self):
        """Test rotating selection with historical data."""
        selector = RotatingStrikeSelector(rotation_period=2)
        
        # Add performance data
        result1 = StrikeSelectionResult(
            call_strike=4530, put_strike=4470, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.MEDIUM_IV, skew_bucket=SkewBucket.MEDIUM_SKEW,
            wing_distance=30, selection_score=0.7, reasoning="Test 1"
        )
        
        result2 = StrikeSelectionResult(
            call_strike=4540, put_strike=4460, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.HIGH_IV, skew_bucket=SkewBucket.LOW_SKEW,
            wing_distance=40, selection_score=0.8, reasoning="Test 2"
        )
        
        # Update performance
        selector.update_performance(result1, 50.0, True, 1000.0)
        selector.update_performance(result2, 100.0, True, 1000.0)
        
        current_price = 4500
        current_iv = 0.5
        current_skew = 0.1
        
        # First selection should use top-ranked combination
        result = selector.select_rotating_strikes(current_price, current_iv, current_skew)
        assert result.iv_percentile == IVPercentile.HIGH_IV
        assert result.skew_bucket == SkewBucket.LOW_SKEW
        
        # Second selection should use second-ranked combination
        result = selector.select_rotating_strikes(current_price, current_iv, current_skew)
        assert result.iv_percentile == IVPercentile.MEDIUM_IV
        assert result.skew_bucket == SkewBucket.MEDIUM_SKEW
    
    def test_get_rotation_status(self):
        """Test getting rotation status."""
        selector = RotatingStrikeSelector()
        
        # No data
        status = selector.get_rotation_status()
        assert status["status"] == "no_data"
        assert status["combinations"] == []
        
        # Add data
        result = StrikeSelectionResult(
            call_strike=4530, put_strike=4470, call_spread=20, put_spread=20,
            iv_percentile=IVPercentile.MEDIUM_IV, skew_bucket=SkewBucket.MEDIUM_SKEW,
            wing_distance=30, selection_score=0.7, reasoning="Test"
        )
        
        selector.update_performance(result, 50.0, True, 1000.0)
        
        status = selector.get_rotation_status()
        assert status["status"] == "rotating"
        assert len(status["combinations"]) == 1
        assert status["current_combination"] == f"{IVPercentile.MEDIUM_IV.value}_{SkewBucket.MEDIUM_SKEW.value}"


class TestStrikeSelectorFactories:
    """Test strike selector factory functions."""
    
    def test_create_default_strike_selector(self):
        """Test creating default strike selector."""
        selector = create_default_strike_selector()
        
        assert isinstance(selector, AdvancedStrikeSelector)
        assert selector.config is not None
    
    def test_create_rotating_strike_selector(self):
        """Test creating rotating strike selector."""
        selector = create_rotating_strike_selector(rotation_period=5)
        
        assert isinstance(selector, RotatingStrikeSelector)
        assert selector.rotation_period == 5
    
    def test_create_custom_strike_selector(self):
        """Test creating custom strike selector."""
        wing_distances = {
            "low_iv": {"low_skew": 10, "medium_skew": 15, "high_skew": 20},
            "medium_iv": {"low_skew": 15, "medium_skew": 25, "high_skew": 35},
            "high_iv": {"low_skew": 20, "medium_skew": 30, "high_skew": 45}
        }
        
        spread_widths = {
            "low_iv": 10,
            "medium_iv": 15,
            "high_iv": 20
        }
        
        selector = create_custom_strike_selector(wing_distances, spread_widths)
        
        assert isinstance(selector, AdvancedStrikeSelector)
        assert selector.config.wing_distances == wing_distances
        assert selector.config.spread_widths == spread_widths


if __name__ == "__main__":
    pytest.main([__file__]) 