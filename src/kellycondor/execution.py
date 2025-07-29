"""
Live execution layer for KellyCondor paper trading through IBKR.
"""

import time
import logging
import redis
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, OrderId

from .processor import Processor
from .sizer import KellySizer
from .entry_windows import (
    EntryWindowManager, 
    WindowAwareProcessor, 
    WindowAwareSizer,
    create_default_window_manager
)
from .strike_selector import (
    AdvancedStrikeSelector,
    RotatingStrikeSelector,
    StrikeSelectionConfig,
    create_default_strike_selector,
    create_rotating_strike_selector
)
from .exit_rules import (
    ExitRuleManager,
    ExitRule,
    ExitTrigger,
    ExitReason,
    ExitDecision,
    create_default_exit_manager,
    create_custom_exit_manager
)
from .regime_analyzer import (
    RegimeAnalyzer,
    RegimeAwareSizer,
    create_regime_analyzer,
    create_regime_aware_sizer
)


@dataclass
class IronCondorOrder:
    """Represents an iron condor order with all legs."""
    symbol: str
    expiry: str
    call_strike: float
    put_strike: float
    call_spread: float
    put_spread: float
    quantity: int
    order_id: Optional[int] = None
    status: str = "PENDING"
    fill_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    window_name: Optional[str] = None


class IBKRClient(EWrapper, EClient):
    """Interactive Brokers API client wrapper."""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
        self.orders = {}
        self.positions = {}
        self.account_info = {}
    
    def isConnected(self):
        """Check if connected to IBKR."""
        return self.connected
        
    def connect_and_run(self, host: str, port: int, client_id: int):
        """Connect to TWS/IB Gateway and start the event loop."""
        try:
            logging.info(f"Attempting to connect to IBKR at {host}:{port}")
            self.connect(host, port, client_id)
            
            # Set a timeout for the connection
            import threading
            import time
            
            def connection_timeout():
                time.sleep(5)  # 5 second timeout
                if not self.connected:
                    logging.error("Connection timeout - IBKR not responding")
                    self.disconnect()
            
            # Start timeout thread
            timeout_thread = threading.Thread(target=connection_timeout)
            timeout_thread.daemon = True
            timeout_thread.start()
            
            # Run the event loop
            self.run()
            
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            self.connected = False
            raise
    
    def nextValidId(self, orderId: int):
        """Callback when next valid order ID is received."""
        self.next_order_id = orderId
        logging.info(f"Next valid order ID: {orderId}")
    
    def orderStatus(self, orderId: int, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Callback for order status updates."""
        if orderId in self.orders:
            self.orders[orderId].status = status
            if status == "Filled":
                self.orders[orderId].fill_price = avgFillPrice
                self.orders[orderId].timestamp = datetime.now()
            logging.info(f"Order {orderId} status: {status}, filled: {filled}, price: {avgFillPrice}")
            
            # Update Redis if we have a Redis client
            try:
                redis_client = redis.Redis(host='localhost', port=6379, db=0)
                trade_key = f"trade:{orderId}"
                if redis_client.exists(trade_key):
                    redis_client.hset(trade_key, 'status', status)
                    if status == "Filled":
                        redis_client.hset(trade_key, 'fill_price', str(avgFillPrice))
                        redis_client.hset(trade_key, 'filled_quantity', str(filled))
            except Exception as e:
                logging.error(f"Failed to update Redis for order {orderId}: {e}")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Callback for position updates."""
        key = f"{contract.symbol}_{contract.secType}_{contract.strike}_{contract.right}"
        self.positions[key] = {
            "position": position,
            "avg_cost": avgCost,
            "contract": contract
        }
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Callback for account summary updates."""
        self.account_info[tag] = {"value": value, "currency": currency}


class LiveIVSkewProcessor(Processor):
    """Live processor for real-time IV and skew calculations."""
    
    def __init__(self, strikes: List[float], expiries: List[str]):
        super().__init__()
        self.strikes = strikes
        self.expiries = expiries
        self.current_iv_data = {}
        self.current_skew = 0.0
        self.current_iv_rank = 0.5
        
    def process_tick(self, tick_data: Dict[str, Any]):
        """Process incoming tick data from market data stream."""
        # Extract IV and skew information from tick
        symbol = tick_data.get("symbol", "SPX")
        strike = tick_data.get("strike")
        expiry = tick_data.get("expiry")
        iv = tick_data.get("iv", 0.0)
        option_type = tick_data.get("type", "C")  # C for call, P for put
        
        if strike and expiry:
            key = f"{symbol}_{strike}_{expiry}_{option_type}"
            self.current_iv_data[key] = iv
            
            # Calculate current skew (put IV - call IV)
            call_key = f"{symbol}_{strike}_{expiry}_C"
            put_key = f"{symbol}_{strike}_{expiry}_P"
            
            if call_key in self.current_iv_data and put_key in self.current_iv_data:
                call_iv = self.current_iv_data[call_key]
                put_iv = self.current_iv_data[put_key]
                self.current_skew = put_iv - call_iv
                
                # Update IV rank (simplified - would need historical data in practice)
                self.current_iv_rank = self._estimate_iv_rank(put_iv)
    
    def _estimate_iv_rank(self, current_iv: float) -> float:
        """Estimate IV rank based on current IV (simplified)."""
        # In practice, this would use historical IV data
        # For now, use a simple heuristic
        if current_iv < 0.15:
            return 0.2  # Low IV
        elif current_iv < 0.25:
            return 0.5  # Medium IV
        else:
            return 0.8  # High IV


class LiveKellySizer(KellySizer):
    """Live Kelly sizer that adapts to real-time market conditions."""
    
    def __init__(self, processor: LiveIVSkewProcessor, win_rate_table: Dict[str, float]):
        super().__init__()
        self.processor = processor
        self.win_rate_table = win_rate_table  # Historical win rates by IV rank
        self.account_size = 100000  # Default, would be updated from IBKR
        
    def get_current_sizing(self) -> Dict[str, float]:
        """Get current position sizing based on live market conditions."""
        iv_rank = self.processor.current_iv_rank
        skew = self.processor.current_skew
        
        # Get win rate for current IV rank
        iv_bucket = self._get_iv_bucket(iv_rank)
        win_rate = self.win_rate_table.get(iv_bucket, 0.6)  # Default 60%
        
        return self.size_position(iv_rank, skew, self.account_size)
    
    def _get_iv_bucket(self, iv_rank: float) -> str:
        """Convert IV rank to bucket for win rate lookup."""
        if iv_rank < 0.3:
            return "low_iv"
        elif iv_rank < 0.7:
            return "medium_iv"
        else:
            return "high_iv"


class IBKRTradeExecutor:
    """Executes iron condor trades through IBKR."""
    
    def __init__(self, client: IBKRClient, processor: Processor, sizer: KellySizer, 
                 window_manager: EntryWindowManager = None, 
                 strike_selector: AdvancedStrikeSelector = None,
                 exit_manager: ExitRuleManager = None,
                 regime_analyzer: RegimeAnalyzer = None):
        self.client = client
        self.processor = processor
        self.sizer = sizer
        self.window_manager = window_manager
        self.strike_selector = strike_selector or create_default_strike_selector()
        self.exit_manager = exit_manager or create_default_exit_manager()
        self.regime_analyzer = regime_analyzer
        self.active_orders = {}
        self.trade_history = []
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Initialize regime-aware sizer if regime analyzer is provided
        if self.regime_analyzer and not isinstance(self.sizer, RegimeAwareSizer):
            self.sizer = create_regime_aware_sizer(self.sizer, self.regime_analyzer)
        
    def update_regime(self, current_volatility: float, current_drift: float):
        """Update the current market regime."""
        if isinstance(self.sizer, RegimeAwareSizer):
            self.sizer.update_regime(current_volatility, current_drift)
            logging.info(f"Updated regime - Volatility: {current_volatility:.3f}, Drift: {current_drift:.3f}")
        
    def submit_iron_condor(self, symbol: str = "SPX", expiry: str = None, 
                          window_name: str = None, current_price: float = None,
                          force_top_ranked: bool = False) -> Optional[IronCondorOrder]:
        """Submit an iron condor order based on current market conditions."""
        if not self.client.connected:
            logging.error("IBKR client not connected")
            return None
            
        # Get current sizing with regime awareness
        if isinstance(self.sizer, RegimeAwareSizer):
            # Update regime based on current market conditions
            current_volatility = getattr(self.processor, 'current_iv_rank', 0.5)
            current_drift = getattr(self.processor, 'current_skew', 0.0)
            self.update_regime(current_volatility, current_drift)
            
            # Get regime-aware sizing
            if isinstance(self.sizer, WindowAwareSizer) and window_name:
                sizing = self.sizer.size_position(
                    self.processor.current_iv_rank,
                    self.processor.current_skew,
                    self.sizer.account_size,
                    window_name
                )
            else:
                sizing = self.sizer.get_current_sizing()
        else:
            # Use standard sizing
            if isinstance(self.sizer, WindowAwareSizer) and window_name:
                sizing = self.sizer.size_position(
                    self.processor.current_iv_rank,
                    self.processor.current_skew,
                    self.sizer.account_size,
                    window_name
                )
            else:
                sizing = self.sizer.get_current_sizing()
        
        # Create iron condor order with advanced strike selection
        order = self._create_iron_condor_order_advanced(symbol, expiry, sizing, current_price, force_top_ranked)
        
        # Submit the order
        success = self._submit_order(order)
        if success:
            self.active_orders[order.order_id] = order
            
            # Add window information to order
            if window_name:
                order.window_name = window_name
            
            # Add regime information to order
            if isinstance(self.sizer, RegimeAwareSizer) and hasattr(self.sizer, 'current_sizing_params'):
                if self.sizer.current_sizing_params:
                    order.regime_type = self.sizer.current_sizing_params.regime_type.value
                    order.regime_confidence = self.sizer.current_sizing_params.confidence
            
            # Add position to exit manager
            self._add_position_to_exit_manager(order)
            
            self._log_trade_to_redis(order)
            
            # Record trade in window manager
            if self.window_manager and window_name:
                trade_size = sizing.get("position_size", 0)
                self.window_manager.record_trade(window_name, 0.0, trade_size)  # PnL will be updated later
            
            return order
        return None
    
    def _add_position_to_exit_manager(self, order: IronCondorOrder):
        """Add a position to the exit manager for tracking."""
        entry_data = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "call_strike": order.call_strike,
            "put_strike": order.put_strike,
            "call_spread": order.call_spread,
            "put_spread": order.put_spread,
            "quantity": order.quantity,
            "entry_iv": getattr(self.processor, 'current_iv_rank', 0.5),
            "entry_skew": getattr(self.processor, 'current_skew', 0.0),
            "entry_time": datetime.now(),
            "expiry_time": datetime.now() + timedelta(days=30),  # Would get from order
            "window_name": order.window_name
        }
        
        if hasattr(order, 'selection_metadata'):
            entry_data.update(order.selection_metadata)
        
        self.exit_manager.add_position(str(order.order_id), entry_data)
        logging.info(f"Added position {order.order_id} to exit manager")
    
    def check_exit_conditions(self, current_data: Dict[str, Any]) -> List[ExitDecision]:
        """Check exit conditions for all active positions."""
        return self.exit_manager.evaluate_exits(current_data)
    
    def close_position(self, order_id: int, exit_decision: ExitDecision):
        """Close a position based on exit decision."""
        if order_id not in self.active_orders:
            logging.warning(f"Order {order_id} not found in active orders")
            return False
        
        order = self.active_orders[order_id]
        
        # Create closing order (opposite of original)
        # This is a simplified version - in practice you'd create proper closing contracts
        try:
            # Log the exit
            logging.info(f"Closing position {order_id} due to {exit_decision.reason.value}: {exit_decision.reasoning}")
            
            # Update order status
            order.status = "CLOSED"
            order.fill_price = 0.0  # Would get from market
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            # Remove from exit manager
            self.exit_manager.remove_position(str(order_id))
            
            # Update Redis
            self._update_trade_in_redis(order, exit_decision)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to close position {order_id}: {e}")
            return False
    
    def _update_trade_in_redis(self, order: IronCondorOrder, exit_decision: ExitDecision):
        """Update trade in Redis with exit information."""
        try:
            trade_key = f"trade:{order.order_id}"
            
            # Update existing trade data
            update_data = {
                "status": "CLOSED",
                "exit_time": datetime.now().isoformat(),
                "exit_reason": exit_decision.reason.value,
                "exit_trigger": exit_decision.trigger.value,
                "exit_reasoning": exit_decision.reasoning,
                "exit_pnl": exit_decision.current_value
            }
            
            self.redis_client.hmset(trade_key, update_data)
            logging.info(f"Updated trade {order.order_id} in Redis with exit data")
            
        except Exception as e:
            logging.error(f"Failed to update trade in Redis: {e}")
    
    def _log_trade_to_redis(self, order: IronCondorOrder):
        """Log trade details to Redis for monitoring."""
        try:
            trade_data = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'expiry': order.expiry,
                'call_strike': order.call_strike,
                'put_strike': order.put_strike,
                'call_spread': order.call_spread,
                'put_spread': order.put_spread,
                'quantity': order.quantity,
                'status': order.status,
                'timestamp': datetime.now().isoformat(),
                'type': 'IRON_CONDOR'
            }
            
            # Save to Redis
            trade_key = f"trade:{order.order_id}"
            self.redis_client.hmset(trade_key, trade_data)
            
            # Also save as position
            position_key = f"position:{order.symbol}_{order.order_id}"
            self.redis_client.hmset(position_key, trade_data)
            
            logging.info(f"Logged trade to Redis: {trade_key}")
            
        except Exception as e:
            logging.error(f"Failed to log trade to Redis: {e}")
    
    def _create_iron_condor_order_advanced(self, symbol: str, expiry: str, sizing: Dict[str, float],
                                          current_price: float = None, force_top_ranked: bool = False) -> IronCondorOrder:
        """Create an iron condor order using advanced strike selection."""
        
        # Get current market price if not provided
        if current_price is None:
            current_price = 4500  # Would get from market data
        
        # Get current IV and skew from processor
        current_iv = getattr(self.processor, 'current_iv_rank', 0.5)
        current_skew = getattr(self.processor, 'current_skew', 0.0)
        
        # Use advanced strike selector
        if isinstance(self.strike_selector, RotatingStrikeSelector):
            # Use rotating selection
            selection_result = self.strike_selector.select_rotating_strikes(
                current_price, current_iv, current_skew
            )
        else:
            # Use standard selection with optional top-ranked forcing
            selection_result = self.strike_selector.select_optimal_strikes(
                current_price, current_iv, current_skew, force_top_ranked=force_top_ranked
            )
        
        # Calculate quantity based on Kelly sizing
        position_size = sizing["position_size"]
        max_risk = sizing["max_risk_amount"]
        quantity = int(position_size / max_risk)  # Simplified
        
        # Create order with selected strikes
        order = IronCondorOrder(
            symbol=symbol,
            expiry=expiry or "20241220",  # Default expiry
            call_strike=selection_result.call_strike,
            put_strike=selection_result.put_strike,
            call_spread=selection_result.call_spread,
            put_spread=selection_result.put_spread,
            quantity=quantity
        )
        
        # Store selection metadata
        order.selection_metadata = {
            "iv_percentile": selection_result.iv_percentile.value,
            "skew_bucket": selection_result.skew_bucket.value,
            "wing_distance": selection_result.wing_distance,
            "selection_score": selection_result.selection_score,
            "reasoning": selection_result.reasoning
        }
        
        return order
    
    def _create_iron_condor_order(self, symbol: str, expiry: str, sizing: Dict[str, float]) -> IronCondorOrder:
        """Create an iron condor order based on current market conditions (legacy method)."""
        # This is the original simplified method - kept for backward compatibility
        current_price = 4500  # Would get from market data
        
        # Iron condor strikes (simplified)
        call_strike = current_price + 50
        put_strike = current_price - 50
        call_spread = 25
        put_spread = 25
        
        # Calculate quantity based on Kelly sizing
        position_size = sizing["position_size"]
        max_risk = sizing["max_risk_amount"]
        quantity = int(position_size / max_risk)  # Simplified
        
        return IronCondorOrder(
            symbol=symbol,
            expiry=expiry or "20241220",  # Default expiry
            call_strike=call_strike,
            put_strike=put_strike,
            call_spread=call_spread,
            put_spread=put_spread,
            quantity=quantity
        )
    
    def _submit_order(self, order: IronCondorOrder) -> bool:
        """Submit the iron condor order to IBKR."""
        try:
            # Create contracts for each leg
            contracts = self._create_iron_condor_contracts(order)
            
            # Create orders for each leg
            orders = self._create_iron_condor_orders(order)
            
            # Submit all orders
            for contract, order_obj in zip(contracts, orders):
                self.client.placeOrder(self.client.next_order_id, contract, order_obj)
                self.client.next_order_id += 1
            
            order.order_id = self.client.next_order_id - len(contracts)
            return True
            
        except Exception as e:
            logging.error(f"Failed to submit order: {e}")
            return False
    
    def _create_iron_condor_contracts(self, order: IronCondorOrder) -> List[Contract]:
        """Create IBKR contracts for iron condor legs."""
        contracts = []
        
        # Sell call spread
        call_short = Contract()
        call_short.symbol = order.symbol
        call_short.secType = "OPT"
        call_short.exchange = "CBOE"
        call_short.currency = "USD"
        call_short.lastTradingDay = order.expiry
        call_short.strike = order.call_strike
        call_short.right = "C"
        contracts.append(call_short)
        
        call_long = Contract()
        call_long.symbol = order.symbol
        call_long.secType = "OPT"
        call_long.exchange = "CBOE"
        call_long.currency = "USD"
        call_long.lastTradingDay = order.expiry
        call_long.strike = order.call_strike + order.call_spread
        call_long.right = "C"
        contracts.append(call_long)
        
        # Sell put spread
        put_short = Contract()
        put_short.symbol = order.symbol
        put_short.secType = "OPT"
        put_short.exchange = "CBOE"
        put_short.currency = "USD"
        put_short.lastTradingDay = order.expiry
        put_short.strike = order.put_strike
        put_short.right = "P"
        contracts.append(put_short)
        
        put_long = Contract()
        put_long.symbol = order.symbol
        put_long.secType = "OPT"
        put_long.exchange = "CBOE"
        put_long.currency = "USD"
        put_long.lastTradingDay = order.expiry
        put_long.strike = order.put_strike - order.put_spread
        put_long.right = "P"
        contracts.append(put_long)
        
        return contracts
    
    def _create_iron_condor_orders(self, order: IronCondorOrder) -> List[Order]:
        """Create IBKR orders for iron condor legs."""
        orders = []
        
        # Sell call spread
        call_short_order = Order()
        call_short_order.action = "SELL"
        call_short_order.totalQuantity = order.quantity
        call_short_order.orderType = "MKT"
        orders.append(call_short_order)
        
        call_long_order = Order()
        call_long_order.action = "BUY"
        call_long_order.totalQuantity = order.quantity
        call_long_order.orderType = "MKT"
        orders.append(call_long_order)
        
        # Sell put spread
        put_short_order = Order()
        put_short_order.action = "SELL"
        put_short_order.totalQuantity = order.quantity
        put_short_order.orderType = "MKT"
        orders.append(put_short_order)
        
        put_long_order = Order()
        put_long_order.action = "BUY"
        put_long_order.totalQuantity = order.quantity
        put_long_order.orderType = "MKT"
        orders.append(put_long_order)
        
        return orders


class DatabentoStream:
    """Simulated market data stream from Databento."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.running = False
        
    def stream_iv_and_skew(self):
        """Stream IV and skew data (simulated)."""
        self.running = True
        while self.running:
            # Simulate tick data
            tick = {
                "symbol": "SPX",
                "strike": 4500,
                "expiry": "20241220",
                "iv": np.random.uniform(0.15, 0.35),
                "type": np.random.choice(["C", "P"]),
                "timestamp": datetime.now()
            }
            yield tick
            time.sleep(1)  # 1 second between ticks
    
    def stop(self):
        """Stop the data stream."""
        self.running = False


class WindowAwareIBKRTradeExecutor(IBKRTradeExecutor):
    """IBKR trade executor with entry window awareness."""
    
    def __init__(self, client: IBKRClient, processor: WindowAwareProcessor, 
                 sizer: WindowAwareSizer, window_manager: EntryWindowManager,
                 strike_selector: AdvancedStrikeSelector = None,
                 exit_manager: ExitRuleManager = None,
                 regime_analyzer: RegimeAnalyzer = None):
        super().__init__(client, processor, sizer, window_manager, strike_selector, exit_manager, regime_analyzer)
        self.window_processor = processor
        self.window_sizer = sizer
        
    def should_trade_now(self, current_time: datetime = None) -> Tuple[bool, Optional[str]]:
        """Check if we should trade at the current time."""
        return self.window_processor.should_trade(current_time)
    
    def get_window_performance_summary(self) -> str:
        """Get a summary of window performance."""
        if self.window_manager:
            return self.window_manager.get_window_summary()
        return "No window manager configured"
    
    def get_window_performance_data(self) -> Dict[str, Dict[str, float]]:
        """Get performance data for all windows."""
        if self.window_manager:
            return self.window_manager.get_all_performance()
        return {}
    
    def get_strike_selection_summary(self) -> str:
        """Get a summary of strike selection performance."""
        if not self.strike_selector:
            return "No strike selector configured"
        
        top_combinations = self.strike_selector.get_top_ranked_combinations()
        
        summary = "Strike Selection Performance:\n"
        summary += "=" * 50 + "\n"
        
        for combination, score in top_combinations:
            performance = self.strike_selector.historical_performance.get(combination, {})
            summary += f"Combination: {combination}\n"
            summary += f"  Score: {score:.3f}\n"
            summary += f"  Trades: {performance.get('trades', 0)}\n"
            summary += f"  Win Rate: {performance.get('win_rate', 0.0):.1%}\n"
            summary += f"  Avg PnL: ${performance.get('avg_pnl', 0.0):.2f}\n"
        
        return summary
    
    def get_exit_rule_summary(self) -> str:
        """Get a summary of exit rule performance."""
        if not self.exit_manager:
            return "No exit manager configured"
        
        return self.exit_manager.get_exit_summary()

    def get_regime_summary(self) -> str:
        """Get a summary of regime analysis."""
        if not self.regime_analyzer:
            return "No regime analyzer configured"
        
        return self.regime_analyzer.get_regime_summary()


def run_paper_trade(api_key: str = None, host: str = "127.0.0.1", port: int = 7497, 
                   client_id: int = 1, simulation_mode: bool = False,
                   enable_windows: bool = True, window_config: List[Dict[str, Any]] = None,
                   enable_advanced_strikes: bool = True, rotation_period: int = 5,
                   force_top_ranked: bool = False, enable_exit_rules: bool = True,
                   exit_config: List[Dict[str, Any]] = None, enable_regime_analysis: bool = True,
                   regime_clusters: int = 6, min_trades_per_regime: int = 10):
    """Main entry point for paper trading with optional entry window, advanced strike selection, exit rules, and regime analysis support."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting KellyCondor paper trading session...")
    
    # Initialize components
    ibkr = IBKRClient()
    
    # Connect to TWS/IB Gateway (skip if simulation mode)
    if not simulation_mode:
        try:
            logger.info(f"Attempting to connect to IBKR at {host}:{port}")
            # Try to connect with a timeout
            import threading
            import time
            
            connection_success = False
            
            def connect_with_timeout():
                nonlocal connection_success
                try:
                    ibkr.connect(host, port, client_id)
                    time.sleep(2)  # Wait for connection
                    if ibkr.isConnected():
                        ibkr.connected = True
                        connection_success = True
                        logger.info("Connected to IBKR")
                    else:
                        logger.error("Connection failed - IBKR not responding")
                except Exception as e:
                    logger.error(f"Connection failed: {e}")
            
            # Start connection in a thread
            connect_thread = threading.Thread(target=connect_with_timeout)
            connect_thread.daemon = True
            connect_thread.start()
            
            # Wait for connection with timeout
            connect_thread.join(timeout=10)
            
            if not connection_success:
                logger.warning("IBKR connection failed, switching to simulation mode")
                simulation_mode = True
                
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            logger.warning("Switching to simulation mode")
            simulation_mode = True
    
    if simulation_mode:
        logger.info("Running in simulation mode - no IBKR connection")
        ibkr.connected = True  # Mock connection for simulation
    
    # Initialize processor and sizer
    strikes = [4400, 4450, 4500, 4550, 4600]
    expiries = ["20241220"]
    base_proc = LiveIVSkewProcessor(strikes, expiries)
    
    # Historical win rates by IV bucket
    win_rate_table = {
        "low_iv": 0.65,
        "medium_iv": 0.60,
        "high_iv": 0.55
    }
    base_sizer = LiveKellySizer(base_proc, win_rate_table)
    
    # Initialize regime analyzer
    regime_analyzer = None
    if enable_regime_analysis:
        regime_analyzer = create_regime_analyzer(
            n_clusters=regime_clusters,
            min_trades_per_cluster=min_trades_per_regime
        )
        logger.info("Regime analysis enabled")
        
        # Load historical trades for regime fitting (in practice, this would come from a database)
        # For now, we'll simulate some historical trades
        historical_trades = _generate_sample_trades(100)
        regime_analyzer.fit_clusters(historical_trades)
        logger.info("Fitted regime clusters to historical data")
    
    # Initialize window manager if enabled
    window_manager = None
    processor = base_proc
    sizer = base_sizer
    
    if enable_windows:
        if window_config:
            from .entry_windows import create_custom_window_manager
            window_manager = create_custom_window_manager(window_config)
        else:
            window_manager = create_default_window_manager()
        
        # Create window-aware processor and sizer
        processor = WindowAwareProcessor(base_proc, window_manager)
        sizer = WindowAwareSizer(base_sizer, window_manager)
        
        logger.info("Entry window management enabled")
        logger.info(window_manager.get_window_summary())
    
    # Initialize strike selector
    strike_selector = None
    if enable_advanced_strikes:
        if rotation_period > 0:
            strike_selector = create_rotating_strike_selector(rotation_period)
            logger.info(f"Advanced strike selection enabled with rotation (period: {rotation_period})")
        else:
            strike_selector = create_default_strike_selector()
            logger.info("Advanced strike selection enabled")
    
    # Initialize exit manager
    exit_manager = None
    if enable_exit_rules:
        if exit_config:
            # Create custom exit rules from configuration
            from .exit_rules import ExitRule, ExitTrigger
            rules = []
            for config in exit_config:
                rule = ExitRule(
                    name=config["name"],
                    trigger=ExitTrigger(config["trigger"]),
                    threshold=config["threshold"],
                    enabled=config.get("enabled", True),
                    priority=config.get("priority", 1)
                )
                # Add specific parameters based on trigger type
                if config["trigger"] == "time_based":
                    rule.time_before_expiry = timedelta(hours=config.get("time_before_expiry_hours", 2))
                elif config["trigger"] == "iv_contraction":
                    rule.iv_contraction_threshold = config.get("iv_contraction_threshold", 0.3)
                elif config["trigger"] == "theta_decay":
                    rule.theta_decay_threshold = config.get("theta_decay_threshold", 0.7)
                elif config["trigger"] == "trailing_pnl":
                    rule.trailing_stop_distance = config.get("trailing_stop_distance", 50)
                    rule.trailing_stop_activation = config.get("trailing_stop_activation", 100)
                    rule.max_profit_take = config.get("max_profit_take", 200)
                    rule.max_loss_stop = config.get("max_loss_stop", -100)
                
                rules.append(rule)
            
            exit_manager = create_custom_exit_manager(rules)
            logger.info(f"Custom exit rules enabled: {len(rules)} rules")
        else:
            exit_manager = create_default_exit_manager()
            logger.info("Default exit rules enabled")
    
    # Initialize executor
    if enable_windows:
        executor = WindowAwareIBKRTradeExecutor(ibkr, processor, sizer, window_manager, strike_selector, exit_manager, regime_analyzer)
    else:
        executor = IBKRTradeExecutor(ibkr, processor, sizer, strike_selector, exit_manager, regime_analyzer)
    
    # Initialize data stream
    stream = DatabentoStream(api_key or "demo_key")
    
    logger.info("Starting market data stream...")
    
    try:
        for tick in stream.stream_iv_and_skew():
            processor.process_tick(tick)
            
            # Update regime analysis with current market conditions
            if enable_regime_analysis and regime_analyzer:
                current_volatility = getattr(processor, 'current_iv_rank', 0.5)
                current_drift = getattr(processor, 'current_skew', 0.0)
                executor.update_regime(current_volatility, current_drift)
            
            # Check exit conditions for existing positions
            if enable_exit_rules and exit_manager:
                current_data = {
                    "current_iv": getattr(processor, 'current_iv_rank', 0.5),
                    "current_skew": getattr(processor, 'current_skew', 0.0),
                    "current_pnl": 0.0,  # Would get from position tracking
                    "timestamp": datetime.now()
                }
                
                exit_decisions = executor.check_exit_conditions(current_data)
                for decision in exit_decisions:
                    # Extract order ID from position ID
                    order_id = int(decision.trigger.value.split("_")[-1]) if "_" in decision.trigger.value else None
                    if order_id:
                        executor.close_position(order_id, decision)
            
            # Check if we should submit a trade (with window awareness)
            if enable_windows:
                should_trade, window_name = executor.should_trade_now()
                if should_trade:
                    order = executor.submit_iron_condor(
                        window_name=window_name,
                        force_top_ranked=force_top_ranked
                    )
                    if order:
                        logger.info(f"Submitted iron condor order in {window_name} window: {order}")
                        if hasattr(order, 'selection_metadata'):
                            logger.info(f"Strike selection: {order.selection_metadata['reasoning']}")
                        if hasattr(order, 'regime_type'):
                            logger.info(f"Regime: {order.regime_type} (confidence: {order.regime_confidence:.2f})")
            else:
                # Original logic without windows
                if _should_submit_trade(processor):
                    order = executor.submit_iron_condor(force_top_ranked=force_top_ranked)
                    if order:
                        logger.info(f"Submitted iron condor order: {order}")
                        if hasattr(order, 'selection_metadata'):
                            logger.info(f"Strike selection: {order.selection_metadata['reasoning']}")
                        if hasattr(order, 'regime_type'):
                            logger.info(f"Regime: {order.regime_type} (confidence: {order.regime_confidence:.2f})")
            
            # Check for order updates
            _check_order_status(executor)
            
            # Log performance summaries periodically
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                if enable_windows and hasattr(executor, 'get_window_performance_summary'):
                    logger.info(executor.get_window_performance_summary())
                
                if enable_advanced_strikes and hasattr(executor, 'get_strike_selection_summary'):
                    logger.info(executor.get_strike_selection_summary())
                
                if enable_exit_rules and hasattr(executor, 'get_exit_rule_summary'):
                    logger.info(executor.get_exit_rule_summary())
                
                if enable_regime_analysis and hasattr(executor, 'get_regime_summary'):
                    logger.info(executor.get_regime_summary())
            
    except KeyboardInterrupt:
        logger.info("Stopping paper trading session...")
        stream.stop()
        ibkr.disconnect()


def _should_submit_trade(processor: LiveIVSkewProcessor) -> bool:
    """Determine if we should submit a trade based on market conditions."""
    # Simple logic - in practice would be more sophisticated
    return processor.current_iv_rank > 0.5 and abs(processor.current_skew) < 0.1


def _check_order_status(executor: IBKRTradeExecutor):
    """Check and log order status updates."""
    for order_id, order in executor.active_orders.items():
        if order.status == "Filled":
            logger.info(f"Order {order_id} filled at {order.fill_price}")
            executor.trade_history.append(order)
            del executor.active_orders[order_id]


def _generate_sample_trades(n_trades: int) -> List[Dict[str, Any]]:
    """Generate sample historical trades for regime analysis."""
    trades = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_trades):
        # Generate trade with some randomness
        np.random.seed(i)  # For reproducibility
        
        # Random trade parameters
        entry_price = 4500 + np.random.normal(0, 50)
        exit_price = entry_price + np.random.normal(0, 100)
        holding_period = np.random.exponential(24)  # Average 24 hours
        pnl = exit_price - entry_price + np.random.normal(0, 50)
        
        trade = {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "holding_period": holding_period,
            "timestamp": base_time + timedelta(hours=i),
            "window": np.random.choice(["morning", "midday", "afternoon"]),
            "iv_rank": np.random.uniform(0.2, 0.8),
            "skew": np.random.uniform(-0.2, 0.2)
        }
        
        trades.append(trade)
    
    return trades


def run_backtest_with_regime_analysis(historical_data: pd.DataFrame, 
                                    window_config: List[Dict[str, Any]] = None,
                                    account_size: float = 100000,
                                    rotation_period: int = 5,
                                    force_top_ranked: bool = False,
                                    exit_config: List[Dict[str, Any]] = None,
                                    regime_clusters: int = 6,
                                    min_trades_per_regime: int = 10) -> Dict[str, Any]:
    """Run backtest with regime analysis, advanced strike selection, entry windows, and exit rules."""
    from .entry_windows import create_custom_window_manager
    from .exit_rules import ExitRuleBacktester, ExitRule, ExitTrigger
    from .regime_analyzer import create_regime_analyzer
    
    # Create regime analyzer
    regime_analyzer = create_regime_analyzer(
        n_clusters=regime_clusters,
        min_trades_per_cluster=min_trades_per_regime
    )
    
    # Generate sample historical trades for regime fitting
    sample_trades = _generate_sample_trades(200)
    regime_analyzer.fit_clusters(sample_trades)
    
    # Create window manager
    if window_config:
        window_manager = create_custom_window_manager(window_config)
    else:
        window_manager = create_default_window_manager()
    
    # Initialize components for backtesting
    strikes = [4400, 4450, 4500, 4550, 4600]
    expiries = ["20241220"]
    processor = LiveIVSkewProcessor(strikes, expiries)
    
    win_rate_table = {
        "low_iv": 0.65,
        "medium_iv": 0.60,
        "high_iv": 0.55
    }
    sizer = LiveKellySizer(processor, win_rate_table)
    
    # Create window-aware components
    window_processor = WindowAwareProcessor(processor, window_manager)
    window_sizer = WindowAwareSizer(sizer, window_manager)
    
    # Create strike selector
    if rotation_period > 0:
        strike_selector = create_rotating_strike_selector(rotation_period)
    else:
        strike_selector = create_default_strike_selector()
    
    # Create exit rule backtester
    backtester = ExitRuleBacktester()
    
    # Add different exit configurations to test
    if exit_config:
        # Test custom exit configuration
        custom_rules = []
        for config in exit_config:
            rule = ExitRule(
                name=config["name"],
                trigger=ExitTrigger(config["trigger"]),
                threshold=config["threshold"],
                enabled=config.get("enabled", True),
                priority=config.get("priority", 1)
            )
            custom_rules.append(rule)
        
        backtester.add_exit_configuration("Custom Exit Rules", custom_rules)
    else:
        # Test default configurations
        from .exit_rules import create_default_exit_manager
        default_manager = create_default_exit_manager()
        backtester.add_exit_configuration("Default Exit Rules", default_manager.rules)
        
        # Test time-based only
        time_only_rules = [
            ExitRule(
                name="Time-Based Only",
                trigger=ExitTrigger.TIME_BASED,
                threshold=0.0,
                time_before_expiry=timedelta(hours=2),
                priority=1
            )
        ]
        backtester.add_exit_configuration("Time-Based Only", time_only_rules)
        
        # Test aggressive exit rules
        aggressive_rules = [
            ExitRule(
                name="Aggressive IV Contraction",
                trigger=ExitTrigger.IV_CONTRACTION,
                threshold=0.2,  # 20% IV contraction
                iv_contraction_threshold=0.2,
                priority=1
            ),
            ExitRule(
                name="Aggressive Theta Decay",
                trigger=ExitTrigger.THETA_DECAY,
                threshold=0.5,  # 50% theta decay
                theta_decay_threshold=0.5,
                priority=2
            ),
            ExitRule(
                name="Tight Trailing Stop",
                trigger=ExitTrigger.TRAILING_PNL,
                threshold=0.0,
                trailing_stop_distance=25,  # 25 points
                trailing_stop_activation=50,  # Activate at $50 profit
                max_profit_take=100,  # Take profit at $100
                max_loss_stop=-50,  # Stop loss at -$50
                priority=3
            )
        ]
        backtester.add_exit_configuration("Aggressive Exit Rules", aggressive_rules)
    
    # Generate entry signals
    entry_signals = []
    current_time = datetime.now()
    
    for _, row in historical_data.iterrows():
        # Simulate entry conditions
        if np.random.random() < 0.1:  # 10% chance of entry
            entry_signal = {
                "price": row.get("price", 4500),
                "iv": row.get("iv", 0.25),
                "timestamp": current_time,
                "expiry_time": current_time + timedelta(days=30)
            }
            entry_signals.append(entry_signal)
        
        current_time += timedelta(hours=1)
    
    # Run backtest
    results = backtester.run_backtest(historical_data, entry_signals)
    
    # Add regime analysis results
    results["regime_analysis"] = {
        "regime_summary": regime_analyzer.get_regime_summary(),
        "best_regime": regime_analyzer.get_best_regime().regime_type.value if regime_analyzer.get_best_regime() else None,
        "total_clusters": len(regime_analyzer.clusters),
        "clusters": {
            cluster_id: {
                "regime_type": cluster.regime_type.value,
                "trade_count": cluster.trade_count,
                "win_rate": cluster.win_rate,
                "sharpe_ratio": cluster.sharpe_ratio,
                "avg_reward": cluster.avg_reward,
                "avg_loss": cluster.avg_loss
            }
            for cluster_id, cluster in regime_analyzer.clusters.items()
        }
    }
    
    # Add additional analysis
    for config_name, config_data in results.items():
        if config_name == "regime_analysis":
            continue
            
        # Calculate additional metrics
        trades = config_data["trades"]
        if trades:
            pnls = [trade["pnl"] for trade in trades]
            config_data["avg_trade_pnl"] = np.mean(pnls)
            config_data["std_trade_pnl"] = np.std(pnls)
            config_data["sharpe_ratio"] = config_data["avg_trade_pnl"] / config_data["std_trade_pnl"] if config_data["std_trade_pnl"] > 0 else 0.0
            config_data["max_profit"] = max(pnls)
            config_data["max_loss"] = min(pnls)
        else:
            config_data["avg_trade_pnl"] = 0.0
            config_data["std_trade_pnl"] = 0.0
            config_data["sharpe_ratio"] = 0.0
            config_data["max_profit"] = 0.0
            config_data["max_loss"] = 0.0
    
    # Add comparison report
    results["comparison_report"] = backtester.compare_configurations()
    
    return results


if __name__ == "__main__":
    run_paper_trade() 