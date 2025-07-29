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
    IronCondorOrder,
    run_paper_trade
)

__all__ = [
    "Processor", 
    "KellySizer", 
    "ReplayEngine",
    "IBKRClient",
    "LiveIVSkewProcessor", 
    "LiveKellySizer",
    "IBKRTradeExecutor",
    "IronCondorOrder",
    "run_paper_trade"
] 