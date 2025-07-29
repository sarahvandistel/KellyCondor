"""
Entry window management for KellyCondor trading strategy.
Supports multiple intraday entry windows with performance tracking.
"""

import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz

from .processor import Processor
from .sizer import KellySizer


class EntryWindowStatus(Enum):
    """Status of an entry window."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    TRADED = "traded"


@dataclass
class EntryWindow:
    """Represents a trading entry window."""
    name: str
    start_time: time
    end_time: time
    timezone: str = "US/Eastern"
    status: EntryWindowStatus = EntryWindowStatus.INACTIVE
    trades_placed: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_trade_size: float = 0.0
    
    def __post_init__(self):
        """Convert timezone string to pytz timezone object."""
        if isinstance(self.timezone, str):
            self.timezone = pytz.timezone(self.timezone)
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if the window is currently active."""
        if self.timezone:
            current_time = current_time.astimezone(self.timezone)
        
        current_time_only = current_time.time()
        return self.start_time <= current_time_only <= self.end_time
    
    def is_expired(self, current_time: datetime) -> bool:
        """Check if the window has expired for today."""
        if self.timezone:
            current_time = current_time.astimezone(self.timezone)
        
        current_time_only = current_time.time()
        return current_time_only > self.end_time
    
    def get_duration_minutes(self) -> int:
        """Get the duration of the window in minutes."""
        start_minutes = self.start_time.hour * 60 + self.start_time.minute
        end_minutes = self.end_time.hour * 60 + self.end_time.minute
        return end_minutes - start_minutes


class EntryWindowManager:
    """Manages multiple entry windows for trading strategy."""
    
    def __init__(self, windows: List[EntryWindow] = None):
        self.windows = windows or self._get_default_windows()
        self.current_window: Optional[EntryWindow] = None
        self.daily_reset_time = time(9, 0)  # 9:00 AM ET
        self.last_reset_date = None
        
    def _get_default_windows(self) -> List[EntryWindow]:
        """Get default entry windows."""
        return [
            EntryWindow("Morning", time(9, 30), time(10, 30)),  # 9:30-10:30 AM ET
            EntryWindow("Mid-Morning", time(11, 0), time(12, 0)),  # 11:00-12:00 PM ET
            EntryWindow("Afternoon", time(14, 0), time(15, 0)),  # 2:00-3:00 PM ET
            EntryWindow("Close", time(15, 30), time(16, 0)),  # 3:30-4:00 PM ET
        ]
    
    def get_active_window(self, current_time: datetime = None) -> Optional[EntryWindow]:
        """Get the currently active entry window."""
        if current_time is None:
            current_time = datetime.now()
        
        # Check if we need to reset daily stats
        self._check_daily_reset(current_time)
        
        for window in self.windows:
            if window.is_active(current_time):
                window.status = EntryWindowStatus.ACTIVE
                self.current_window = window
                return window
            elif window.is_expired(current_time):
                window.status = EntryWindowStatus.EXPIRED
            else:
                window.status = EntryWindowStatus.INACTIVE
        
        self.current_window = None
        return None
    
    def _check_daily_reset(self, current_time: datetime):
        """Reset daily statistics if it's a new trading day."""
        if self.last_reset_date is None:
            self.last_reset_date = current_time.date()
            return
        
        if current_time.date() > self.last_reset_date:
            # New trading day - reset window statistics
            for window in self.windows:
                window.trades_placed = 0
                window.total_pnl = 0.0
                window.win_rate = 0.0
                window.avg_trade_size = 0.0
                window.status = EntryWindowStatus.INACTIVE
            
            self.last_reset_date = current_time.date()
            logging.info("Daily window statistics reset")
    
    def record_trade(self, window_name: str, pnl: float, trade_size: float):
        """Record a trade for a specific window."""
        for window in self.windows:
            if window.name == window_name:
                window.trades_placed += 1
                window.total_pnl += pnl
                window.avg_trade_size = (
                    (window.avg_trade_size * (window.trades_placed - 1) + trade_size) 
                    / window.trades_placed
                )
                
                # Update win rate (simplified - assumes positive PnL = win)
                if window.trades_placed > 0:
                    wins = sum(1 for _ in range(window.trades_placed) if pnl > 0)
                    window.win_rate = wins / window.trades_placed
                
                logging.info(f"Recorded trade for {window_name}: PnL={pnl:.2f}, Size={trade_size:.2f}")
                break
    
    def get_window_performance(self, window_name: str) -> Dict[str, float]:
        """Get performance metrics for a specific window."""
        for window in self.windows:
            if window.name == window_name:
                return {
                    "trades_placed": window.trades_placed,
                    "total_pnl": window.total_pnl,
                    "win_rate": window.win_rate,
                    "avg_trade_size": window.avg_trade_size,
                    "avg_pnl_per_trade": window.total_pnl / window.trades_placed if window.trades_placed > 0 else 0.0
                }
        return {}
    
    def get_all_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all windows."""
        return {window.name: self.get_window_performance(window.name) for window in self.windows}
    
    def is_trading_allowed(self, current_time: datetime = None) -> bool:
        """Check if trading is allowed at the current time."""
        active_window = self.get_active_window(current_time)
        return active_window is not None
    
    def get_next_window(self, current_time: datetime = None) -> Optional[EntryWindow]:
        """Get the next upcoming entry window."""
        if current_time is None:
            current_time = datetime.now()
        
        for window in self.windows:
            if window.start_time > current_time.time():
                return window
        
        return None  # No more windows today
    
    def get_window_summary(self) -> str:
        """Get a summary of all windows and their performance."""
        summary = "Entry Windows Summary:\n"
        summary += "=" * 50 + "\n"
        
        for window in self.windows:
            status_icon = "🟢" if window.status == EntryWindowStatus.ACTIVE else "⚪"
            summary += f"{status_icon} {window.name}: {window.start_time.strftime('%H:%M')}-{window.end_time.strftime('%H:%M')}\n"
            summary += f"   Trades: {window.trades_placed}, PnL: ${window.total_pnl:.2f}, Win Rate: {window.win_rate:.1%}\n"
        
        return summary


class WindowAwareProcessor(Processor):
    """Processor that considers entry windows for trading decisions."""
    
    def __init__(self, base_processor: Processor, window_manager: EntryWindowManager):
        super().__init__()
        self.base_processor = base_processor
        self.window_manager = window_manager
        self.last_trade_time = None
        self.min_trade_interval = timedelta(minutes=30)  # Minimum time between trades
    
    def process_tick(self, tick_data: Dict[str, Any]):
        """Process tick data with window awareness."""
        # Delegate to base processor
        self.base_processor.process_tick(tick_data)
        
        # Add window information to tick data
        current_time = datetime.now()
        active_window = self.window_manager.get_active_window(current_time)
        
        if active_window:
            tick_data["entry_window"] = active_window.name
            tick_data["window_active"] = True
        else:
            tick_data["entry_window"] = None
            tick_data["window_active"] = False
    
    def should_trade(self, current_time: datetime = None) -> Tuple[bool, Optional[str]]:
        """Determine if we should trade based on window and market conditions."""
        if current_time is None:
            current_time = datetime.now()
        
        # Check if we're in an active window
        active_window = self.window_manager.get_active_window(current_time)
        if not active_window:
            return False, None
        
        # Check minimum trade interval
        if (self.last_trade_time and 
            current_time - self.last_trade_time < self.min_trade_interval):
            return False, None
        
        # Check base processor conditions (delegate to base processor)
        # This would typically check IV rank, skew, etc.
        base_conditions_met = self._check_base_conditions()
        
        if base_conditions_met:
            self.last_trade_time = current_time
            return True, active_window.name
        
        return False, None
    
    def _check_base_conditions(self) -> bool:
        """Check base trading conditions (IV rank, skew, etc.)."""
        # This would implement the logic from the base processor
        # For now, return True to allow trading when in active windows
        return True


class WindowAwareSizer(KellySizer):
    """Kelly sizer that considers entry window performance."""
    
    def __init__(self, base_sizer: KellySizer, window_manager: EntryWindowManager):
        super().__init__()
        self.base_sizer = base_sizer
        self.window_manager = window_manager
        self.window_adjustment_factors = {
            "Morning": 1.0,      # No adjustment
            "Mid-Morning": 0.9,   # Slightly reduce size
            "Afternoon": 0.8,     # Reduce size more
            "Close": 0.7,         # Most conservative
        }
    
    def size_position(self, iv_rank: float, skew: float, account_size: float, 
                     window_name: str = None) -> Dict[str, float]:
        """Size position with window-specific adjustments."""
        # Get base sizing from the underlying sizer
        base_sizing = self.base_sizer.size_position(iv_rank, skew, account_size)
        
        # Apply window-specific adjustments
        if window_name and window_name in self.window_adjustment_factors:
            adjustment_factor = self.window_adjustment_factors[window_name]
            
            # Adjust position size based on window performance
            window_performance = self.window_manager.get_window_performance(window_name)
            if window_performance["win_rate"] < 0.5:
                # Reduce size if window has poor performance
                adjustment_factor *= 0.8
            
            # Apply adjustments
            base_sizing["position_size"] *= adjustment_factor
            base_sizing["max_risk_amount"] *= adjustment_factor
            
            logging.info(f"Window {window_name} adjustment: {adjustment_factor:.2f}")
        
        return base_sizing


def create_default_window_manager() -> EntryWindowManager:
    """Create a default window manager with common trading windows."""
    windows = [
        EntryWindow("Morning", time(9, 30), time(10, 30)),
        EntryWindow("Mid-Morning", time(11, 0), time(12, 0)),
        EntryWindow("Afternoon", time(14, 0), time(15, 0)),
        EntryWindow("Close", time(15, 30), time(16, 0)),
    ]
    return EntryWindowManager(windows)


def create_custom_window_manager(window_configs: List[Dict[str, any]]) -> EntryWindowManager:
    """Create a custom window manager from configuration."""
    windows = []
    for config in window_configs:
        window = EntryWindow(
            name=config["name"],
            start_time=time.fromisoformat(config["start_time"]),
            end_time=time.fromisoformat(config["end_time"]),
            timezone=config.get("timezone", "US/Eastern")
        )
        windows.append(window)
    
    return EntryWindowManager(windows) 