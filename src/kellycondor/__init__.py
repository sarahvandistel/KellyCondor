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
    run_backtest_with_regime_analysis,
    run_historical_backtest_with_databento
)
from .entry_windows import (
    EntryWindow,
    EntryWindowManager,
    WindowAwareProcessor,
    WindowAwareSizer,
    create_default_window_manager,
    create_custom_window_manager
)
from .strike_selector import (
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
from .exit_rules import (
    ExitRuleManager,
    ExitRule,
    ExitTrigger,
    ExitReason,
    ExitDecision,
    ExitRuleBacktester,
    create_default_exit_manager,
    create_custom_exit_manager,
    create_exit_backtester
)
from .regime_analyzer import (
    RegimeAnalyzer,
    RegimeAwareSizer,
    RegimeCluster,
    RegimeSizingParams,
    RegimeType,
    create_regime_analyzer,
    create_regime_aware_sizer
)
from .databento_historical import (
    DatabentoHistoricalData,
    HistoricalDataConfig,
    MarketDataPoint,
    create_historical_data_client,
    get_backtest_data
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
    "run_backtest_with_regime_analysis",
    "run_historical_backtest_with_databento",
    "EntryWindow",
    "EntryWindowManager", 
    "WindowAwareProcessor",
    "WindowAwareSizer",
    "create_default_window_manager",
    "create_custom_window_manager",
    "AdvancedStrikeSelector",
    "RotatingStrikeSelector",
    "StrikeSelectionConfig",
    "StrikeSelectionResult",
    "SkewBucket",
    "IVPercentile",
    "create_default_strike_selector",
    "create_rotating_strike_selector",
    "create_custom_strike_selector",
    "ExitRuleManager",
    "ExitRule",
    "ExitTrigger",
    "ExitReason",
    "ExitDecision",
    "ExitRuleBacktester",
    "create_default_exit_manager",
    "create_custom_exit_manager",
    "create_exit_backtester",
    "RegimeAnalyzer",
    "RegimeAwareSizer",
    "RegimeCluster",
    "RegimeSizingParams",
    "RegimeType",
    "create_regime_analyzer",
    "create_regime_aware_sizer",
    "DatabentoHistoricalData",
    "HistoricalDataConfig",
    "MarketDataPoint",
    "create_historical_data_client",
    "get_backtest_data"
] 