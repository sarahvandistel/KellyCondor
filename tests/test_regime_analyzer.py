"""
Tests for regime analysis functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from kellycondor.regime_analyzer import (
    RegimeAnalyzer,
    RegimeAwareSizer,
    RegimeCluster,
    RegimeSizingParams,
    RegimeType,
    create_regime_analyzer,
    create_regime_aware_sizer
)


class TestRegimeType:
    """Test RegimeType enum."""
    
    def test_regime_type_values(self):
        """Test regime type enum values."""
        assert RegimeType.LOW_VOL_LOW_DRIFT.value == "low_vol_low_drift"
        assert RegimeType.LOW_VOL_HIGH_DRIFT.value == "low_vol_high_drift"
        assert RegimeType.MEDIUM_VOL_LOW_DRIFT.value == "medium_vol_low_drift"
        assert RegimeType.MEDIUM_VOL_HIGH_DRIFT.value == "medium_vol_high_drift"
        assert RegimeType.HIGH_VOL_LOW_DRIFT.value == "high_vol_low_drift"
        assert RegimeType.HIGH_VOL_HIGH_DRIFT.value == "high_vol_high_drift"


class TestRegimeCluster:
    """Test RegimeCluster dataclass."""
    
    def test_regime_cluster_creation(self):
        """Test creating a regime cluster."""
        cluster = RegimeCluster(
            cluster_id=0,
            regime_type=RegimeType.MEDIUM_VOL_LOW_DRIFT,
            center_volatility=0.25,
            center_drift=0.02,
            trade_count=50,
            win_rate=0.65,
            avg_reward=150.0,
            avg_loss=-100.0,
            profit_factor=1.5,
            sharpe_ratio=0.8,
            max_drawdown=-200.0,
            avg_trade_duration=24.0,
            volatility_range=(0.20, 0.30),
            drift_range=(-0.05, 0.05)
        )
        
        assert cluster.cluster_id == 0
        assert cluster.regime_type == RegimeType.MEDIUM_VOL_LOW_DRIFT
        assert cluster.center_volatility == 0.25
        assert cluster.center_drift == 0.02
        assert cluster.trade_count == 50
        assert cluster.win_rate == 0.65
        assert cluster.avg_reward == 150.0
        assert cluster.avg_loss == -100.0
        assert cluster.profit_factor == 1.5
        assert cluster.sharpe_ratio == 0.8
        assert cluster.max_drawdown == -200.0
        assert cluster.avg_trade_duration == 24.0
        assert cluster.volatility_range == (0.20, 0.30)
        assert cluster.drift_range == (-0.05, 0.05)
        assert cluster.trades == []


class TestRegimeSizingParams:
    """Test RegimeSizingParams dataclass."""
    
    def test_regime_sizing_params_creation(self):
        """Test creating regime sizing parameters."""
        params = RegimeSizingParams(
            regime_type=RegimeType.HIGH_VOL_HIGH_DRIFT,
            kelly_fraction=0.15,
            max_position_size=0.8,
            risk_per_trade=0.02,
            win_rate=0.55,
            avg_reward=120.0,
            avg_loss=-90.0,
            confidence=0.85
        )
        
        assert params.regime_type == RegimeType.HIGH_VOL_HIGH_DRIFT
        assert params.kelly_fraction == 0.15
        assert params.max_position_size == 0.8
        assert params.risk_per_trade == 0.02
        assert params.win_rate == 0.55
        assert params.avg_reward == 120.0
        assert params.avg_loss == -90.0
        assert params.confidence == 0.85


class TestRegimeAnalyzer:
    """Test RegimeAnalyzer class."""
    
    def test_analyzer_creation(self):
        """Test creating a regime analyzer."""
        analyzer = RegimeAnalyzer(n_clusters=6, min_trades_per_cluster=10)
        
        assert analyzer.n_clusters == 6
        assert analyzer.min_trades_per_cluster == 10
        assert analyzer.clusters == {}
        assert analyzer.is_fitted is False
        assert analyzer.kmeans is None
    
    def test_calculate_realized_volatility(self):
        """Test realized volatility calculation."""
        analyzer = RegimeAnalyzer()
        
        # Test with simple price series
        price_series = [100, 101, 99, 102, 98, 103, 97, 104]
        volatility = analyzer.calculate_realized_volatility(price_series, window=3)
        
        assert isinstance(volatility, float)
        assert volatility >= 0.0
    
    def test_calculate_directional_drift(self):
        """Test directional drift calculation."""
        analyzer = RegimeAnalyzer()
        
        # Test with simple price series
        price_series = [100, 101, 102, 103, 104, 105, 106, 107]  # Upward trend
        drift = analyzer.calculate_directional_drift(price_series, window=5)
        
        assert isinstance(drift, float)
        # Should be positive for upward trend
        assert drift > 0.0
    
    def test_extract_trade_features(self):
        """Test extracting features from a trade."""
        analyzer = RegimeAnalyzer()
        
        trade = {
            "entry_price": 4500,
            "exit_price": 4520,
            "holding_period": 24,
            "timestamp": datetime.now()
        }
        
        volatility, drift = analyzer.extract_trade_features(trade)
        
        assert isinstance(volatility, float)
        assert isinstance(drift, float)
        assert volatility >= 0.0
    
    def test_determine_regime_type(self):
        """Test regime type determination."""
        analyzer = RegimeAnalyzer()
        
        # Test low volatility, low drift
        regime = analyzer.determine_regime_type(0.10, -0.02)
        assert regime == RegimeType.LOW_VOL_LOW_DRIFT
        
        # Test high volatility, high drift
        regime = analyzer.determine_regime_type(0.35, 0.08)
        assert regime == RegimeType.HIGH_VOL_HIGH_DRIFT
        
        # Test medium volatility, low drift
        regime = analyzer.determine_regime_type(0.25, -0.03)
        assert regime == RegimeType.MEDIUM_VOL_LOW_DRIFT
    
    def test_fit_clusters(self):
        """Test fitting clusters to trade data."""
        analyzer = RegimeAnalyzer(n_clusters=3, min_trades_per_cluster=5)
        
        # Generate sample trades
        trades = []
        for i in range(30):
            trade = {
                "entry_price": 4500 + np.random.normal(0, 50),
                "exit_price": 4500 + np.random.normal(0, 100),
                "pnl": np.random.normal(50, 100),
                "holding_period": np.random.exponential(24),
                "timestamp": datetime.now() - timedelta(days=i),
                "window": "morning",
                "iv_rank": np.random.uniform(0.2, 0.8),
                "skew": np.random.uniform(-0.2, 0.2)
            }
            trades.append(trade)
        
        clusters = analyzer.fit_clusters(trades)
        
        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        assert analyzer.is_fitted is True
        assert analyzer.kmeans is not None
        
        # Check that each cluster has the required minimum trades
        for cluster_id, cluster in clusters.items():
            assert cluster.trade_count >= analyzer.min_trades_per_cluster
            assert cluster.regime_type in RegimeType
    
    def test_predict_regime(self):
        """Test regime prediction."""
        analyzer = RegimeAnalyzer(n_clusters=3, min_trades_per_cluster=5)
        
        # First fit clusters
        trades = []
        for i in range(30):
            trade = {
                "entry_price": 4500 + np.random.normal(0, 50),
                "exit_price": 4500 + np.random.normal(0, 100),
                "pnl": np.random.normal(50, 100),
                "holding_period": np.random.exponential(24),
                "timestamp": datetime.now() - timedelta(days=i),
                "window": "morning",
                "iv_rank": np.random.uniform(0.2, 0.8),
                "skew": np.random.uniform(-0.2, 0.2)
            }
            trades.append(trade)
        
        analyzer.fit_clusters(trades)
        
        # Test prediction
        predicted_cluster = analyzer.predict_regime(0.25, 0.02)
        
        assert predicted_cluster is not None
        assert isinstance(predicted_cluster, RegimeCluster)
        assert predicted_cluster.cluster_id in analyzer.clusters
    
    def test_get_regime_sizing_params(self):
        """Test getting sizing parameters for a regime."""
        analyzer = RegimeAnalyzer()
        
        # Create a sample cluster
        cluster = RegimeCluster(
            cluster_id=0,
            regime_type=RegimeType.MEDIUM_VOL_LOW_DRIFT,
            center_volatility=0.25,
            center_drift=0.02,
            trade_count=50,
            win_rate=0.65,
            avg_reward=150.0,
            avg_loss=-100.0,
            profit_factor=1.5,
            sharpe_ratio=0.8,
            max_drawdown=-200.0,
            avg_trade_duration=24.0,
            volatility_range=(0.20, 0.30),
            drift_range=(-0.05, 0.05)
        )
        
        params = analyzer.get_regime_sizing_params(cluster)
        
        assert isinstance(params, RegimeSizingParams)
        assert params.regime_type == cluster.regime_type
        assert params.win_rate == cluster.win_rate
        assert params.avg_reward == cluster.avg_reward
        assert params.avg_loss == cluster.avg_loss
        assert 0.0 <= params.kelly_fraction <= 0.25
        assert params.confidence > 0.0
    
    def test_get_regime_summary(self):
        """Test getting regime summary."""
        analyzer = RegimeAnalyzer()
        
        # Test with no clusters
        summary = analyzer.get_regime_summary()
        assert "No regimes fitted yet" in summary
        
        # Test with clusters
        analyzer.clusters = {
            0: RegimeCluster(
                cluster_id=0,
                regime_type=RegimeType.MEDIUM_VOL_LOW_DRIFT,
                center_volatility=0.25,
                center_drift=0.02,
                trade_count=50,
                win_rate=0.65,
                avg_reward=150.0,
                avg_loss=-100.0,
                profit_factor=1.5,
                sharpe_ratio=0.8,
                max_drawdown=-200.0,
                avg_trade_duration=24.0,
                volatility_range=(0.20, 0.30),
                drift_range=(-0.05, 0.05)
            )
        }
        
        summary = analyzer.get_regime_summary()
        assert "Market Regime Analysis" in summary
        assert "Regime 0" in summary
        assert "medium_vol_low_drift" in summary
    
    def test_get_best_regime(self):
        """Test getting the best performing regime."""
        analyzer = RegimeAnalyzer()
        
        # Test with no clusters
        best_regime = analyzer.get_best_regime()
        assert best_regime is None
        
        # Test with clusters
        analyzer.clusters = {
            0: RegimeCluster(
                cluster_id=0,
                regime_type=RegimeType.MEDIUM_VOL_LOW_DRIFT,
                center_volatility=0.25,
                center_drift=0.02,
                trade_count=50,
                win_rate=0.65,
                avg_reward=150.0,
                avg_loss=-100.0,
                profit_factor=1.5,
                sharpe_ratio=0.8,
                max_drawdown=-200.0,
                avg_trade_duration=24.0,
                volatility_range=(0.20, 0.30),
                drift_range=(-0.05, 0.05)
            ),
            1: RegimeCluster(
                cluster_id=1,
                regime_type=RegimeType.HIGH_VOL_HIGH_DRIFT,
                center_volatility=0.35,
                center_drift=0.08,
                trade_count=30,
                win_rate=0.45,
                avg_reward=100.0,
                avg_loss=-120.0,
                profit_factor=0.8,
                sharpe_ratio=0.3,
                max_drawdown=-300.0,
                avg_trade_duration=18.0,
                volatility_range=(0.30, 0.40),
                drift_range=(0.05, 0.10)
            )
        }
        
        best_regime = analyzer.get_best_regime()
        assert best_regime is not None
        # Should be the one with higher Sharpe ratio and win rate
        assert best_regime.cluster_id == 0


class TestRegimeAwareSizer:
    """Test RegimeAwareSizer class."""
    
    def test_sizer_creation(self):
        """Test creating a regime-aware sizer."""
        # Mock base sizer
        base_sizer = Mock()
        base_sizer.get_current_sizing.return_value = {
            "position_size": 1000,
            "max_risk_amount": 100
        }
        base_sizer.size_position.return_value = {
            "position_size": 1000,
            "max_risk_amount": 100
        }
        
        # Mock regime analyzer
        regime_analyzer = Mock()
        
        sizer = RegimeAwareSizer(base_sizer, regime_analyzer)
        
        assert sizer.base_sizer == base_sizer
        assert sizer.regime_analyzer == regime_analyzer
        assert sizer.current_regime is None
        assert sizer.current_sizing_params is None
    
    def test_update_regime(self):
        """Test updating the current regime."""
        # Mock base sizer
        base_sizer = Mock()
        
        # Mock regime analyzer
        regime_analyzer = Mock()
        mock_cluster = Mock()
        mock_cluster.regime_type.value = "medium_vol_low_drift"
        regime_analyzer.predict_regime.return_value = mock_cluster
        
        mock_params = RegimeSizingParams(
            regime_type=RegimeType.MEDIUM_VOL_LOW_DRIFT,
            kelly_fraction=0.15,
            max_position_size=1.0,
            risk_per_trade=0.025,
            win_rate=0.65,
            avg_reward=150.0,
            avg_loss=-100.0,
            confidence=0.8
        )
        regime_analyzer.get_regime_sizing_params.return_value = mock_params
        
        sizer = RegimeAwareSizer(base_sizer, regime_analyzer)
        
        sizer.update_regime(0.25, 0.02)
        
        assert sizer.current_regime == mock_cluster
        assert sizer.current_sizing_params == mock_params
        regime_analyzer.predict_regime.assert_called_once_with(0.25, 0.02)
    
    def test_get_current_sizing_with_regime(self):
        """Test getting current sizing with regime awareness."""
        # Mock base sizer
        base_sizer = Mock()
        base_sizer.get_current_sizing.return_value = {
            "position_size": 1000,
            "max_risk_amount": 100
        }
        
        # Mock regime analyzer
        regime_analyzer = Mock()
        
        # Mock sizing parameters
        mock_params = RegimeSizingParams(
            regime_type=RegimeType.MEDIUM_VOL_LOW_DRIFT,
            kelly_fraction=0.15,
            max_position_size=1.0,
            risk_per_trade=0.025,
            win_rate=0.65,
            avg_reward=150.0,
            avg_loss=-100.0,
            confidence=0.8
        )
        
        sizer = RegimeAwareSizer(base_sizer, regime_analyzer)
        sizer.current_sizing_params = mock_params
        
        sizing = sizer.get_current_sizing()
        
        assert "position_size" in sizing
        assert "max_risk_amount" in sizing
        assert "kelly_fraction" in sizing
        assert "regime_type" in sizing
        assert "regime_confidence" in sizing
        assert sizing["regime_type"] == "medium_vol_low_drift"
        assert sizing["regime_confidence"] == 0.8
    
    def test_get_current_sizing_without_regime(self):
        """Test getting current sizing without regime."""
        # Mock base sizer
        base_sizer = Mock()
        base_sizer.get_current_sizing.return_value = {
            "position_size": 1000,
            "max_risk_amount": 100
        }
        
        # Mock regime analyzer
        regime_analyzer = Mock()
        
        sizer = RegimeAwareSizer(base_sizer, regime_analyzer)
        sizer.current_sizing_params = None
        
        sizing = sizer.get_current_sizing()
        
        assert sizing == base_sizer.get_current_sizing.return_value
    
    def test_size_position_with_regime(self):
        """Test sizing position with regime awareness."""
        # Mock base sizer
        base_sizer = Mock()
        base_sizer.size_position.return_value = {
            "position_size": 1000,
            "max_risk_amount": 100
        }
        
        # Mock regime analyzer
        regime_analyzer = Mock()
        
        # Mock sizing parameters
        mock_params = RegimeSizingParams(
            regime_type=RegimeType.HIGH_VOL_HIGH_DRIFT,
            kelly_fraction=0.12,
            max_position_size=0.8,
            risk_per_trade=0.02,
            win_rate=0.55,
            avg_reward=120.0,
            avg_loss=-90.0,
            confidence=0.7
        )
        
        sizer = RegimeAwareSizer(base_sizer, regime_analyzer)
        sizer.current_sizing_params = mock_params
        
        sizing = sizer.size_position(0.3, 0.05, 100000, "morning")
        
        assert "position_size" in sizing
        assert "max_risk_amount" in sizing
        assert "kelly_fraction" in sizing
        assert "regime_type" in sizing
        assert "regime_confidence" in sizing
        assert "window_name" in sizing
        assert sizing["regime_type"] == "high_vol_high_drift"
        assert sizing["regime_confidence"] == 0.7
        assert sizing["window_name"] == "morning"
    
    def test_size_position_without_regime(self):
        """Test sizing position without regime."""
        # Mock base sizer
        base_sizer = Mock()
        base_sizer.size_position.return_value = {
            "position_size": 1000,
            "max_risk_amount": 100
        }
        
        # Mock regime analyzer
        regime_analyzer = Mock()
        
        sizer = RegimeAwareSizer(base_sizer, regime_analyzer)
        sizer.current_sizing_params = None
        
        sizing = sizer.size_position(0.3, 0.05, 100000, "morning")
        
        assert sizing == base_sizer.size_position.return_value


class TestRegimeAnalyzerFactories:
    """Test regime analyzer factory functions."""
    
    def test_create_regime_analyzer(self):
        """Test creating a regime analyzer."""
        analyzer = create_regime_analyzer(n_clusters=8, min_trades_per_cluster=15)
        
        assert isinstance(analyzer, RegimeAnalyzer)
        assert analyzer.n_clusters == 8
        assert analyzer.min_trades_per_cluster == 15
    
    def test_create_regime_aware_sizer(self):
        """Test creating a regime-aware sizer."""
        # Mock base sizer and regime analyzer
        base_sizer = Mock()
        regime_analyzer = Mock()
        
        sizer = create_regime_aware_sizer(base_sizer, regime_analyzer)
        
        assert isinstance(sizer, RegimeAwareSizer)
        assert sizer.base_sizer == base_sizer
        assert sizer.regime_analyzer == regime_analyzer


if __name__ == "__main__":
    pytest.main([__file__]) 