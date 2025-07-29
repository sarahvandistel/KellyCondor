"""
Regime analysis for KellyCondor strategy.
Clusters historical trades by realized volatility and directional drift,
then provides regime-specific sizing parameters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class RegimeType(Enum):
    """Market regime types based on volatility and drift."""
    LOW_VOL_LOW_DRIFT = "low_vol_low_drift"
    LOW_VOL_HIGH_DRIFT = "low_vol_high_drift"
    MEDIUM_VOL_LOW_DRIFT = "medium_vol_low_drift"
    MEDIUM_VOL_HIGH_DRIFT = "medium_vol_high_drift"
    HIGH_VOL_LOW_DRIFT = "high_vol_low_drift"
    HIGH_VOL_HIGH_DRIFT = "high_vol_high_drift"


@dataclass
class RegimeCluster:
    """A cluster of trades representing a market regime."""
    cluster_id: int
    regime_type: RegimeType
    center_volatility: float
    center_drift: float
    trade_count: int
    win_rate: float
    avg_reward: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    volatility_range: Tuple[float, float]
    drift_range: Tuple[float, float]
    trades: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []


@dataclass
class RegimeSizingParams:
    """Sizing parameters specific to a market regime."""
    regime_type: RegimeType
    kelly_fraction: float
    max_position_size: float
    risk_per_trade: float
    win_rate: float
    avg_reward: float
    avg_loss: float
    confidence: float  # How confident we are in this regime's parameters


class RegimeAnalyzer:
    """Analyzes historical trades to identify market regimes and their characteristics."""
    
    def __init__(self, n_clusters: int = 6, min_trades_per_cluster: int = 10):
        self.n_clusters = n_clusters
        self.min_trades_per_cluster = min_trades_per_cluster
        self.clusters: Dict[int, RegimeCluster] = {}
        self.scaler = StandardScaler()
        self.kmeans = None
        self.is_fitted = False
        
    def calculate_realized_volatility(self, price_series: List[float], window: int = 20) -> float:
        """Calculate realized volatility from price series."""
        if len(price_series) < window + 1:
            return 0.0
        
        returns = np.diff(np.log(price_series))
        if len(returns) < window:
            return np.std(returns) * np.sqrt(252)  # Annualized
        
        # Rolling volatility
        rolling_vol = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            rolling_vol.append(vol)
        
        return np.mean(rolling_vol) if rolling_vol else 0.0
    
    def calculate_directional_drift(self, price_series: List[float], window: int = 20) -> float:
        """Calculate directional drift from price series."""
        if len(price_series) < window + 1:
            return 0.0
        
        # Calculate drift as the slope of a linear regression
        x = np.arange(len(price_series))
        y = np.array(price_series)
        
        # Use last 'window' points for drift calculation
        if len(price_series) > window:
            x = x[-window:]
            y = y[-window:]
        
        # Linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        # Convert to annualized drift
        annualized_drift = slope * 252 / len(price_series)
        
        return annualized_drift
    
    def extract_trade_features(self, trade: Dict[str, Any]) -> Tuple[float, float]:
        """Extract volatility and drift features from a trade."""
        # Get price series during trade
        entry_price = trade.get("entry_price", 4500)
        exit_price = trade.get("exit_price", 4500)
        
        # Simulate price series during trade (in practice, this would come from market data)
        trade_duration = trade.get("holding_period", 24)  # hours
        n_points = max(10, int(trade_duration / 2))  # At least 10 points
        
        # Generate price series with some randomness
        np.random.seed(hash(str(trade.get("timestamp", datetime.now())))  # For reproducibility
        price_changes = np.random.normal(0, 0.01, n_points)  # 1% daily volatility
        price_series = [entry_price]
        
        for change in price_changes:
            new_price = price_series[-1] * (1 + change)
            price_series.append(new_price)
        
        # Ensure exit price is close to the last price
        price_series[-1] = exit_price
        
        # Calculate features
        volatility = self.calculate_realized_volatility(price_series)
        drift = self.calculate_directional_drift(price_series)
        
        return volatility, drift
    
    def prepare_trade_data(self, trades: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Prepare trade data for clustering."""
        features = []
        processed_trades = []
        
        for trade in trades:
            try:
                volatility, drift = self.extract_trade_features(trade)
                
                # Add features
                features.append([volatility, drift])
                
                # Add processed trade with features
                processed_trade = trade.copy()
                processed_trade["realized_volatility"] = volatility
                processed_trade["directional_drift"] = drift
                processed_trades.append(processed_trade)
                
            except Exception as e:
                logging.warning(f"Failed to process trade: {e}")
                continue
        
        if not features:
            raise ValueError("No valid trades found for clustering")
        
        return np.array(features), processed_trades
    
    def determine_regime_type(self, volatility: float, drift: float) -> RegimeType:
        """Determine regime type based on volatility and drift values."""
        # Define thresholds (these could be made configurable)
        vol_low_threshold = 0.15
        vol_high_threshold = 0.30
        drift_low_threshold = -0.05
        drift_high_threshold = 0.05
        
        # Determine volatility level
        if volatility < vol_low_threshold:
            vol_level = "low"
        elif volatility > vol_high_threshold:
            vol_level = "high"
        else:
            vol_level = "medium"
        
        # Determine drift level
        if drift < drift_low_threshold:
            drift_level = "low"
        elif drift > drift_high_threshold:
            drift_level = "high"
        else:
            drift_level = "medium"
        
        # Map to regime type
        regime_map = {
            ("low", "low"): RegimeType.LOW_VOL_LOW_DRIFT,
            ("low", "medium"): RegimeType.LOW_VOL_LOW_DRIFT,
            ("low", "high"): RegimeType.LOW_VOL_HIGH_DRIFT,
            ("medium", "low"): RegimeType.MEDIUM_VOL_LOW_DRIFT,
            ("medium", "medium"): RegimeType.MEDIUM_VOL_LOW_DRIFT,
            ("medium", "high"): RegimeType.MEDIUM_VOL_HIGH_DRIFT,
            ("high", "low"): RegimeType.HIGH_VOL_LOW_DRIFT,
            ("high", "medium"): RegimeType.HIGH_VOL_LOW_DRIFT,
            ("high", "high"): RegimeType.HIGH_VOL_HIGH_DRIFT,
        }
        
        return regime_map.get((vol_level, drift_level), RegimeType.MEDIUM_VOL_LOW_DRIFT)
    
    def fit_clusters(self, trades: List[Dict[str, Any]]) -> Dict[int, RegimeCluster]:
        """Fit clusters to historical trade data."""
        logging.info(f"Fitting regime clusters to {len(trades)} trades")
        
        # Prepare data
        features, processed_trades = self.prepare_trade_data(trades)
        
        if len(features) < self.n_clusters:
            logging.warning(f"Not enough trades ({len(features)}) for {self.n_clusters} clusters")
            self.n_clusters = min(len(features), 3)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            logging.info(f"Silhouette score: {silhouette_avg:.3f}")
        
        # Create clusters
        self.clusters = {}
        cluster_centers = self.kmeans.cluster_centers_
        
        for cluster_id in range(self.n_clusters):
            # Get trades for this cluster
            cluster_trades = [trade for i, trade in enumerate(processed_trades) 
                            if cluster_labels[i] == cluster_id]
            
            if len(cluster_trades) < self.min_trades_per_cluster:
                logging.warning(f"Cluster {cluster_id} has only {len(cluster_trades)} trades, skipping")
                continue
            
            # Calculate cluster statistics
            volatilities = [trade["realized_volatility"] for trade in cluster_trades]
            drifts = [trade["directional_drift"] for trade in cluster_trades]
            pnls = [trade.get("pnl", 0.0) for trade in cluster_trades]
            
            # Center in original space
            center_scaled = cluster_centers[cluster_id]
            center_original = self.scaler.inverse_transform([center_scaled])[0]
            center_volatility, center_drift = center_original
            
            # Calculate performance metrics
            wins = sum(1 for pnl in pnls if pnl > 0)
            win_rate = wins / len(pnls) if pnls else 0.0
            avg_reward = np.mean([pnl for pnl in pnls if pnl > 0]) if any(pnl > 0 for pnl in pnls) else 0.0
            avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if any(pnl < 0 for pnl in pnls) else 0.0
            
            # Calculate profit factor
            total_profit = sum(pnl for pnl in pnls if pnl > 0)
            total_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            returns = [pnl / 1000 for pnl in pnls]  # Normalize by position size
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            
            # Calculate max drawdown
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            # Calculate average trade duration
            durations = [trade.get("holding_period", 24) for trade in cluster_trades]
            avg_trade_duration = np.mean(durations) if durations else 24.0
            
            # Determine regime type
            regime_type = self.determine_regime_type(center_volatility, center_drift)
            
            # Create cluster
            cluster = RegimeCluster(
                cluster_id=cluster_id,
                regime_type=regime_type,
                center_volatility=center_volatility,
                center_drift=center_drift,
                trade_count=len(cluster_trades),
                win_rate=win_rate,
                avg_reward=avg_reward,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_trade_duration=avg_trade_duration,
                volatility_range=(min(volatilities), max(volatilities)),
                drift_range=(min(drifts), max(drifts)),
                trades=cluster_trades
            )
            
            self.clusters[cluster_id] = cluster
            
            logging.info(f"Cluster {cluster_id} ({regime_type.value}): "
                        f"{len(cluster_trades)} trades, "
                        f"Win rate: {win_rate:.1%}, "
                        f"Avg reward: ${avg_reward:.2f}, "
                        f"Sharpe: {sharpe_ratio:.3f}")
        
        self.is_fitted = True
        return self.clusters
    
    def predict_regime(self, current_volatility: float, current_drift: float) -> Optional[RegimeCluster]:
        """Predict the current market regime."""
        if not self.is_fitted or not self.kmeans:
            return None
        
        # Scale features
        features = np.array([[current_volatility, current_drift]])
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        cluster_id = self.kmeans.predict(features_scaled)[0]
        
        return self.clusters.get(cluster_id)
    
    def get_regime_sizing_params(self, regime_cluster: RegimeCluster) -> RegimeSizingParams:
        """Get sizing parameters for a specific regime."""
        if not regime_cluster:
            return None
        
        # Calculate Kelly fraction based on win rate and reward/loss ratio
        win_rate = regime_cluster.win_rate
        avg_reward = regime_cluster.avg_reward
        avg_loss = abs(regime_cluster.avg_loss)
        
        if avg_loss == 0:
            kelly_fraction = 0.1  # Conservative default
        else:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            b = avg_reward / avg_loss if avg_loss > 0 else 1.0
            p = win_rate
            q = 1 - win_rate
            kelly_fraction = (b * p - q) / b if b > 0 else 0.0
        
        # Apply constraints
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))  # Between 0% and 25%
        
        # Calculate confidence based on cluster size and performance consistency
        confidence = min(1.0, regime_cluster.trade_count / 50)  # More trades = higher confidence
        
        # Adjust sizing based on regime characteristics
        if regime_cluster.regime_type in [RegimeType.HIGH_VOL_HIGH_DRIFT, RegimeType.HIGH_VOL_LOW_DRIFT]:
            # High volatility regimes: reduce position size
            max_position_size = 0.8
            risk_per_trade = 0.02  # 2% risk per trade
        elif regime_cluster.regime_type in [RegimeType.LOW_VOL_LOW_DRIFT, RegimeType.LOW_VOL_HIGH_DRIFT]:
            # Low volatility regimes: can increase position size
            max_position_size = 1.2
            risk_per_trade = 0.03  # 3% risk per trade
        else:
            # Medium volatility regimes: standard sizing
            max_position_size = 1.0
            risk_per_trade = 0.025  # 2.5% risk per trade
        
        return RegimeSizingParams(
            regime_type=regime_cluster.regime_type,
            kelly_fraction=kelly_fraction,
            max_position_size=max_position_size,
            risk_per_trade=risk_per_trade,
            win_rate=win_rate,
            avg_reward=avg_reward,
            avg_loss=avg_loss,
            confidence=confidence
        )
    
    def get_regime_summary(self) -> str:
        """Get a summary of all regimes."""
        if not self.clusters:
            return "No regimes fitted yet"
        
        summary = "Market Regime Analysis:\n"
        summary += "=" * 60 + "\n\n"
        
        for cluster_id, cluster in self.clusters.items():
            summary += f"Regime {cluster_id} ({cluster.regime_type.value}):\n"
            summary += f"  Volatility: {cluster.center_volatility:.3f} ({cluster.volatility_range[0]:.3f} - {cluster.volatility_range[1]:.3f})\n"
            summary += f"  Drift: {cluster.center_drift:.3f} ({cluster.drift_range[0]:.3f} - {cluster.drift_range[1]:.3f})\n"
            summary += f"  Trades: {cluster.trade_count}\n"
            summary += f"  Win Rate: {cluster.win_rate:.1%}\n"
            summary += f"  Avg Reward: ${cluster.avg_reward:.2f}\n"
            summary += f"  Avg Loss: ${cluster.avg_loss:.2f}\n"
            summary += f"  Profit Factor: {cluster.profit_factor:.2f}\n"
            summary += f"  Sharpe Ratio: {cluster.sharpe_ratio:.3f}\n"
            summary += f"  Max Drawdown: ${cluster.max_drawdown:.2f}\n"
            summary += f"  Avg Duration: {cluster.avg_trade_duration:.1f} hours\n\n"
        
        return summary
    
    def get_best_regime(self) -> Optional[RegimeCluster]:
        """Get the best performing regime."""
        if not self.clusters:
            return None
        
        # Score regimes by Sharpe ratio and win rate
        best_cluster = None
        best_score = -float('inf')
        
        for cluster in self.clusters.values():
            if cluster.trade_count < self.min_trades_per_cluster:
                continue
            
            # Composite score: Sharpe ratio + win rate bonus
            score = cluster.sharpe_ratio + (cluster.win_rate * 0.5)
            
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        return best_cluster


class RegimeAwareSizer:
    """Kelly sizer that adapts to market regimes."""
    
    def __init__(self, base_sizer, regime_analyzer: RegimeAnalyzer):
        self.base_sizer = base_sizer
        self.regime_analyzer = regime_analyzer
        self.current_regime = None
        self.current_sizing_params = None
        
    def update_regime(self, current_volatility: float, current_drift: float):
        """Update the current regime based on market conditions."""
        self.current_regime = self.regime_analyzer.predict_regime(current_volatility, current_drift)
        
        if self.current_regime:
            self.current_sizing_params = self.regime_analyzer.get_regime_sizing_params(self.current_regime)
            logging.info(f"Current regime: {self.current_regime.regime_type.value} "
                        f"(confidence: {self.current_sizing_params.confidence:.2f})")
        else:
            self.current_sizing_params = None
            logging.warning("No regime detected, using base sizer")
    
    def get_current_sizing(self) -> Dict[str, float]:
        """Get current sizing parameters with regime awareness."""
        if not self.current_sizing_params:
            return self.base_sizer.get_current_sizing()
        
        # Get base sizing
        base_sizing = self.base_sizer.get_current_sizing()
        
        # Apply regime-specific adjustments
        regime_params = self.current_sizing_params
        
        # Adjust position size based on regime
        adjusted_position_size = base_sizing.get("position_size", 1000) * regime_params.max_position_size
        
        # Adjust risk based on regime
        adjusted_risk = base_sizing.get("max_risk_amount", 100) * regime_params.risk_per_trade
        
        # Apply Kelly fraction from regime
        kelly_adjusted_size = adjusted_position_size * regime_params.kelly_fraction
        
        return {
            "position_size": kelly_adjusted_size,
            "max_risk_amount": adjusted_risk,
            "kelly_fraction": regime_params.kelly_fraction,
            "regime_type": regime_params.regime_type.value,
            "regime_confidence": regime_params.confidence,
            "regime_win_rate": regime_params.win_rate,
            "regime_avg_reward": regime_params.avg_reward
        }
    
    def size_position(self, iv_rank: float, skew: float, account_size: float, 
                     window_name: str = None) -> Dict[str, float]:
        """Size position with regime awareness."""
        if not self.current_sizing_params:
            return self.base_sizer.size_position(iv_rank, skew, account_size, window_name)
        
        # Get base sizing
        base_sizing = self.base_sizer.size_position(iv_rank, skew, account_size, window_name)
        
        # Apply regime-specific adjustments
        regime_params = self.current_sizing_params
        
        # Adjust position size based on regime
        adjusted_position_size = base_sizing.get("position_size", 1000) * regime_params.max_position_size
        
        # Apply Kelly fraction from regime
        kelly_adjusted_size = adjusted_position_size * regime_params.kelly_fraction
        
        return {
            "position_size": kelly_adjusted_size,
            "max_risk_amount": base_sizing.get("max_risk_amount", 100) * regime_params.risk_per_trade,
            "kelly_fraction": regime_params.kelly_fraction,
            "regime_type": regime_params.regime_type.value,
            "regime_confidence": regime_params.confidence,
            "regime_win_rate": regime_params.win_rate,
            "regime_avg_reward": regime_params.avg_reward,
            "window_name": window_name
        }


def create_regime_analyzer(n_clusters: int = 6, min_trades_per_cluster: int = 10) -> RegimeAnalyzer:
    """Create a regime analyzer."""
    return RegimeAnalyzer(n_clusters=n_clusters, min_trades_per_cluster=min_trades_per_cluster)


def create_regime_aware_sizer(base_sizer, regime_analyzer: RegimeAnalyzer) -> RegimeAwareSizer:
    """Create a regime-aware sizer."""
    return RegimeAwareSizer(base_sizer, regime_analyzer) 