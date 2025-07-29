"""
Advanced strike selection for iron condor strategies based on IV percentile and skew buckets.
Supports parameterized wing distances and dynamic strike selection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SkewBucket(Enum):
    """Skew buckets for strike selection."""
    LOW_SKEW = "low_skew"      # < 0.05
    MEDIUM_SKEW = "medium_skew"  # 0.05 - 0.15
    HIGH_SKEW = "high_skew"    # > 0.15


class IVPercentile(Enum):
    """IV percentile buckets for strike selection."""
    LOW_IV = "low_iv"          # < 30th percentile
    MEDIUM_IV = "medium_iv"    # 30th - 70th percentile
    HIGH_IV = "high_iv"        # > 70th percentile


@dataclass
class StrikeSelectionConfig:
    """Configuration for strike selection strategy."""
    # Wing distance configurations (in points)
    wing_distances: Dict[str, Dict[str, int]] = None
    
    # IV percentile thresholds
    iv_low_threshold: float = 0.3
    iv_high_threshold: float = 0.7
    
    # Skew thresholds
    skew_low_threshold: float = 0.05
    skew_high_threshold: float = 0.15
    
    # Minimum and maximum wing distances
    min_wing_distance: int = 10
    max_wing_distance: int = 100
    
    # Spread width configurations
    spread_widths: Dict[str, int] = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.wing_distances is None:
            self.wing_distances = {
                "low_iv": {
                    "low_skew": 15,
                    "medium_skew": 20,
                    "high_skew": 25
                },
                "medium_iv": {
                    "low_skew": 20,
                    "medium_skew": 30,
                    "high_skew": 40
                },
                "high_iv": {
                    "low_skew": 25,
                    "medium_skew": 35,
                    "high_skew": 50
                }
            }
        
        if self.spread_widths is None:
            self.spread_widths = {
                "low_iv": 15,
                "medium_iv": 20,
                "high_iv": 25
            }


@dataclass
class StrikeSelectionResult:
    """Result of strike selection process."""
    call_strike: float
    put_strike: float
    call_spread: float
    put_spread: float
    iv_percentile: IVPercentile
    skew_bucket: SkewBucket
    wing_distance: int
    selection_score: float
    reasoning: str


class AdvancedStrikeSelector:
    """Advanced strike selector using IV percentile and skew buckets."""
    
    def __init__(self, config: StrikeSelectionConfig = None):
        self.config = config or StrikeSelectionConfig()
        self.historical_performance = {}
        self.selection_history = []
        
    def get_iv_percentile(self, current_iv: float, historical_ivs: List[float]) -> IVPercentile:
        """Determine IV percentile bucket."""
        if not historical_ivs:
            return IVPercentile.MEDIUM_IV
        
        iv_percentile = np.percentile(historical_ivs, 100 * current_iv)
        
        if iv_percentile < self.config.iv_low_threshold:
            return IVPercentile.LOW_IV
        elif iv_percentile > self.config.iv_high_threshold:
            return IVPercentile.HIGH_IV
        else:
            return IVPercentile.MEDIUM_IV
    
    def get_skew_bucket(self, current_skew: float) -> SkewBucket:
        """Determine skew bucket."""
        if abs(current_skew) < self.config.skew_low_threshold:
            return SkewBucket.LOW_SKEW
        elif abs(current_skew) > self.config.skew_high_threshold:
            return SkewBucket.HIGH_SKEW
        else:
            return SkewBucket.MEDIUM_SKEW
    
    def get_wing_distance(self, iv_percentile: IVPercentile, skew_bucket: SkewBucket) -> int:
        """Get wing distance based on IV percentile and skew bucket."""
        iv_key = iv_percentile.value
        skew_key = skew_bucket.value
        
        if iv_key in self.config.wing_distances and skew_key in self.config.wing_distances[iv_key]:
            return self.config.wing_distances[iv_key][skew_key]
        
        # Default to medium values if not found
        return self.config.wing_distances["medium_iv"]["medium_skew"]
    
    def get_spread_width(self, iv_percentile: IVPercentile) -> int:
        """Get spread width based on IV percentile."""
        iv_key = iv_percentile.value
        return self.config.spread_widths.get(iv_key, 20)
    
    def calculate_strikes(self, current_price: float, iv_percentile: IVPercentile, 
                         skew_bucket: SkewBucket) -> StrikeSelectionResult:
        """Calculate optimal strikes based on market conditions."""
        
        # Get wing distance and spread width
        wing_distance = self.get_wing_distance(iv_percentile, skew_bucket)
        spread_width = self.get_spread_width(iv_percentile)
        
        # Calculate strikes
        call_strike = current_price + wing_distance
        put_strike = current_price - wing_distance
        
        # Calculate spreads
        call_spread = spread_width
        put_spread = spread_width
        
        # Calculate selection score (higher is better)
        selection_score = self._calculate_selection_score(iv_percentile, skew_bucket, wing_distance)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(iv_percentile, skew_bucket, wing_distance, spread_width)
        
        return StrikeSelectionResult(
            call_strike=call_strike,
            put_strike=put_strike,
            call_spread=call_spread,
            put_spread=put_spread,
            iv_percentile=iv_percentile,
            skew_bucket=skew_bucket,
            wing_distance=wing_distance,
            selection_score=selection_score,
            reasoning=reasoning
        )
    
    def _calculate_selection_score(self, iv_percentile: IVPercentile, skew_bucket: SkewBucket, 
                                 wing_distance: int) -> float:
        """Calculate selection score based on historical performance."""
        key = f"{iv_percentile.value}_{skew_bucket.value}"
        
        if key in self.historical_performance:
            performance = self.historical_performance[key]
            return performance.get("win_rate", 0.5) * performance.get("avg_pnl", 0.0)
        
        # Default scoring based on market conditions
        base_score = 0.5
        
        # Adjust for IV percentile
        if iv_percentile == IVPercentile.MEDIUM_IV:
            base_score += 0.1
        elif iv_percentile == IVPercentile.HIGH_IV:
            base_score += 0.05
        
        # Adjust for skew
        if skew_bucket == SkewBucket.MEDIUM_SKEW:
            base_score += 0.1
        elif skew_bucket == SkewBucket.HIGH_SKEW:
            base_score += 0.05
        
        # Adjust for wing distance (prefer moderate distances)
        if 20 <= wing_distance <= 40:
            base_score += 0.1
        elif wing_distance > 50:
            base_score -= 0.1
        
        return min(max(base_score, 0.0), 1.0)
    
    def _generate_reasoning(self, iv_percentile: IVPercentile, skew_bucket: SkewBucket,
                           wing_distance: int, spread_width: int) -> str:
        """Generate reasoning for strike selection."""
        reasoning_parts = []
        
        # IV reasoning
        if iv_percentile == IVPercentile.LOW_IV:
            reasoning_parts.append("Low IV environment - using tighter wings")
        elif iv_percentile == IVPercentile.HIGH_IV:
            reasoning_parts.append("High IV environment - using wider wings")
        else:
            reasoning_parts.append("Medium IV environment - using standard wings")
        
        # Skew reasoning
        if skew_bucket == SkewBucket.LOW_SKEW:
            reasoning_parts.append("Low skew - balanced strike selection")
        elif skew_bucket == SkewBucket.HIGH_SKEW:
            reasoning_parts.append("High skew - adjusted for volatility asymmetry")
        else:
            reasoning_parts.append("Medium skew - standard strike selection")
        
        # Wing distance reasoning
        reasoning_parts.append(f"Wing distance: {wing_distance} points")
        reasoning_parts.append(f"Spread width: {spread_width} points")
        
        return " | ".join(reasoning_parts)
    
    def update_performance(self, selection_result: StrikeSelectionResult, pnl: float, 
                          win: bool, trade_size: float):
        """Update historical performance for strike selection."""
        key = f"{selection_result.iv_percentile.value}_{selection_result.skew_bucket.value}"
        
        if key not in self.historical_performance:
            self.historical_performance[key] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "total_size": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_size": 0.0
            }
        
        perf = self.historical_performance[key]
        perf["trades"] += 1
        perf["total_pnl"] += pnl
        perf["total_size"] += trade_size
        
        if win:
            perf["wins"] += 1
        
        # Update averages
        perf["win_rate"] = perf["wins"] / perf["trades"]
        perf["avg_pnl"] = perf["total_pnl"] / perf["trades"]
        perf["avg_size"] = perf["total_size"] / perf["trades"]
        
        # Log the update
        logging.info(f"Updated performance for {key}: Win rate={perf['win_rate']:.2f}, "
                    f"Avg PnL=${perf['avg_pnl']:.2f}")
    
    def get_top_ranked_combinations(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get top-ranked IV/skew combinations based on historical performance."""
        combinations = []
        
        for key, performance in self.historical_performance.items():
            if performance["trades"] >= 3:  # Minimum sample size
                score = performance["win_rate"] * performance["avg_pnl"]
                combinations.append((key, score))
        
        # Sort by score (descending) and return top combinations
        combinations.sort(key=lambda x: x[1], reverse=True)
        return combinations[:limit]
    
    def select_optimal_strikes(self, current_price: float, current_iv: float, 
                              current_skew: float, historical_ivs: List[float] = None,
                              force_top_ranked: bool = False) -> StrikeSelectionResult:
        """Select optimal strikes using advanced criteria."""
        
        # Determine IV percentile and skew bucket
        iv_percentile = self.get_iv_percentile(current_iv, historical_ivs or [])
        skew_bucket = self.get_skew_bucket(current_skew)
        
        # If forcing top-ranked combinations, check if current combination is in top ranks
        if force_top_ranked:
            top_combinations = self.get_top_ranked_combinations()
            current_key = f"{iv_percentile.value}_{skew_bucket.value}"
            
            # Check if current combination is in top ranks
            top_keys = [combo[0] for combo in top_combinations]
            if current_key not in top_keys and top_combinations:
                # Use the top-ranked combination instead
                best_key = top_combinations[0][0]
                iv_part, skew_part = best_key.split("_", 1)
                
                # Map back to enums
                iv_percentile = IVPercentile(iv_part)
                skew_bucket = SkewBucket(skew_part)
                
                logging.info(f"Forcing top-ranked combination: {best_key}")
        
        # Calculate strikes
        result = self.calculate_strikes(current_price, iv_percentile, skew_bucket)
        
        # Log selection
        logging.info(f"Strike selection: {result.reasoning}")
        logging.info(f"Call: {result.call_strike}, Put: {result.put_strike}, "
                    f"Score: {result.selection_score:.3f}")
        
        return result


class RotatingStrikeSelector(AdvancedStrikeSelector):
    """Strike selector that rotates through top-ranked combinations."""
    
    def __init__(self, config: StrikeSelectionConfig = None, rotation_period: int = 5):
        super().__init__(config)
        self.rotation_period = rotation_period
        self.current_rotation_index = 0
        self.rotation_history = []
    
    def select_rotating_strikes(self, current_price: float, current_iv: float,
                               current_skew: float, historical_ivs: List[float] = None) -> StrikeSelectionResult:
        """Select strikes using rotating top-ranked combinations."""
        
        # Get top-ranked combinations
        top_combinations = self.get_top_ranked_combinations()
        
        if not top_combinations:
            # Fall back to standard selection
            return self.select_optimal_strikes(current_price, current_iv, current_skew, historical_ivs)
        
        # Rotate through top combinations
        selected_key = top_combinations[self.current_rotation_index % len(top_combinations)][0]
        iv_part, skew_part = selected_key.split("_", 1)
        
        # Map to enums
        iv_percentile = IVPercentile(iv_part)
        skew_bucket = SkewBucket(skew_part)
        
        # Calculate strikes
        result = self.calculate_strikes(current_price, iv_percentile, skew_bucket)
        
        # Update rotation index
        self.current_rotation_index += 1
        if self.current_rotation_index >= len(top_combinations) * self.rotation_period:
            self.current_rotation_index = 0
        
        # Log rotation
        logging.info(f"Rotating to combination {selected_key} (rotation {self.current_rotation_index})")
        
        return result
    
    def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status."""
        top_combinations = self.get_top_ranked_combinations()
        
        if not top_combinations:
            return {"status": "no_data", "combinations": []}
        
        current_key = top_combinations[self.current_rotation_index % len(top_combinations)][0]
        
        return {
            "status": "rotating",
            "current_combination": current_key,
            "rotation_index": self.current_rotation_index,
            "total_combinations": len(top_combinations),
            "combinations": top_combinations
        }


def create_default_strike_selector() -> AdvancedStrikeSelector:
    """Create a default strike selector with standard configuration."""
    config = StrikeSelectionConfig()
    return AdvancedStrikeSelector(config)


def create_rotating_strike_selector(rotation_period: int = 5) -> RotatingStrikeSelector:
    """Create a rotating strike selector."""
    config = StrikeSelectionConfig()
    return RotatingStrikeSelector(config, rotation_period)


def create_custom_strike_selector(wing_distances: Dict[str, Dict[str, int]],
                                spread_widths: Dict[str, int] = None) -> AdvancedStrikeSelector:
    """Create a custom strike selector with specific configurations."""
    config = StrikeSelectionConfig(
        wing_distances=wing_distances,
        spread_widths=spread_widths
    )
    return AdvancedStrikeSelector(config) 