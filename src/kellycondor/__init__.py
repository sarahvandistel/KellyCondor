"""
KellyCondor - A live-paper-ready SPX 0DTE iron-condor engine using Kelly-criterion-based sizing.
"""

__version__ = "0.1.0"
__author__ = "Sarah"

from .processor import Processor
from .sizer import KellySizer
from .replay import ReplayEngine
from .execution import (
    IBKRClient, 
    LiveIVSkewProcessor, 
    LiveKellySizer, 
    IBKRTradeExecutor,
    WindowAwareIBKRTradeExecutor,
    IronCondorOrder,
    run_paper_trade,
    run_backtest_with_windows
)
from .entry_windows import (
    EntryWindow,
    EntryWindowManager,
    WindowAwareProcessor,
    WindowAwareSizer,
    create_default_window_manager,
    create_custom_window_manager
)

__all__ = [
    "Processor", 
    "KellySizer", 
    "ReplayEngine",
    "IBKRClient",
    "LiveIVSkewProcessor", 
    "LiveKellySizer",
    "IBKRTradeExecutor",
    "WindowAwareIBKRTradeExecutor",
    "IronCondorOrder",
    "run_paper_trade",
    "run_backtest_with_windows",
    "EntryWindow",
    "EntryWindowManager", 
    "WindowAwareProcessor",
    "WindowAwareSizer",
    "create_default_window_manager",
    "create_custom_window_manager"
] 