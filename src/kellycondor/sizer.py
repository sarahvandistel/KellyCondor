"""
Kelly criterion-based position sizer for SPX iron condors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class KellySizer:
    """Kelly criterion-based position sizer for iron condor strategies."""
    
    def __init__(self, max_kelly_fraction: float = 0.25):
        self.max_kelly_fraction = max_kelly_fraction
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly fraction based on win rate and payout ratio.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss == 0:
            return 0.0
            
        payout_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * payout_ratio - (1 - win_rate)) / payout_ratio
        
        # Cap at maximum allowed fraction
        return min(max(kelly_fraction, 0.0), self.max_kelly_fraction)
    
    def size_position(self, 
                     iv_rank: float, 
                     skew: float, 
                     account_size: float,
                     max_risk_per_trade: float = 0.02) -> Dict[str, float]:
        """
        Size iron condor position based on market conditions.
        
        Args:
            iv_rank: IV Rank (0-1)
            skew: Volatility skew
            account_size: Total account size
            max_risk_per_trade: Maximum risk per trade as fraction of account
            
        Returns:
            Dictionary with position sizing parameters
        """
        # Adjust Kelly fraction based on IV Rank and skew
        base_kelly = 0.15  # Base Kelly fraction
        
        # IV Rank adjustment: higher IV = more conservative
        iv_adjustment = 1.0 - (iv_rank * 0.3)
        
        # Skew adjustment: higher skew = more conservative
        skew_adjustment = 1.0 - (abs(skew) * 0.1)
        
        adjusted_kelly = base_kelly * iv_adjustment * skew_adjustment
        adjusted_kelly = min(adjusted_kelly, self.max_kelly_fraction)
        
        # Calculate position size
        max_risk_amount = account_size * max_risk_per_trade
        position_size = max_risk_amount * adjusted_kelly
        
        return {
            "kelly_fraction": adjusted_kelly,
            "position_size": position_size,
            "max_risk_amount": max_risk_amount,
            "iv_rank": iv_rank,
            "skew": skew
        } 