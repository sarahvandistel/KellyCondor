"""
Live execution layer for KellyCondor paper trading through IBKR.
"""

import time
import logging
import redis
import json
from typing import Dict, Any, Optional, List
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


class IBKRClient(EWrapper, EClient):
    """Interactive Brokers API client wrapper."""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
        self.orders = {}
        self.positions = {}
        self.account_info = {}
        
    def connect_and_run(self, host: str, port: int, client_id: int):
        """Connect to TWS/IB Gateway and start the event loop."""
        self.connect(host, port, client_id)
        self.connected = True
        self.run()
    
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
    
    def __init__(self, client: IBKRClient, processor: LiveIVSkewProcessor, sizer: LiveKellySizer):
        self.client = client
        self.processor = processor
        self.sizer = sizer
        self.active_orders = {}
        self.trade_history = []
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def submit_iron_condor(self, symbol: str = "SPX", expiry: str = None) -> Optional[IronCondorOrder]:
        """Submit an iron condor order based on current market conditions."""
        if not self.client.connected:
            logging.error("IBKR client not connected")
            return None
            
        # Get current sizing
        sizing = self.sizer.get_current_sizing()
        
        # Create iron condor order
        order = self._create_iron_condor_order(symbol, expiry, sizing)
        
        # Submit the order
        success = self._submit_order(order)
        if success:
            self.active_orders[order.order_id] = order
            self._log_trade_to_redis(order)
            return order
        return None
    
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
    
    def _create_iron_condor_order(self, symbol: str, expiry: str, sizing: Dict[str, float]) -> IronCondorOrder:
        """Create an iron condor order based on current market conditions."""
        # Calculate strikes based on current market conditions
        # This is a simplified version - in practice you'd use current SPX price
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


def run_paper_trade(api_key: str = None, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1, simulation_mode: bool = False):
    """Main entry point for paper trading."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting KellyCondor paper trading session...")
    
    # Initialize components
    ibkr = IBKRClient()
    
    # Connect to TWS/IB Gateway (skip if simulation mode)
    if not simulation_mode:
        try:
            ibkr.connect_and_run(host, port, client_id)
            logger.info("Connected to IBKR")
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return
    else:
        logger.info("Running in simulation mode - no IBKR connection")
        ibkr.connected = True  # Mock connection for simulation
    
    # Initialize processor and sizer
    strikes = [4400, 4450, 4500, 4550, 4600]
    expiries = ["20241220"]
    proc = LiveIVSkewProcessor(strikes, expiries)
    
    # Historical win rates by IV bucket
    win_rate_table = {
        "low_iv": 0.65,
        "medium_iv": 0.60,
        "high_iv": 0.55
    }
    sizer = LiveKellySizer(proc, win_rate_table)
    
    # Initialize executor
    executor = IBKRTradeExecutor(ibkr, proc, sizer)
    
    # Initialize data stream
    stream = DatabentoStream(api_key or "demo_key")
    
    logger.info("Starting market data stream...")
    
    try:
        for tick in stream.stream_iv_and_skew():
            proc.process_tick(tick)
            
            # Check if we should submit a trade
            if _should_submit_trade(proc):
                order = executor.submit_iron_condor()
                if order:
                    logger.info(f"Submitted iron condor order: {order}")
            
            # Check for order updates
            _check_order_status(executor)
            
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


if __name__ == "__main__":
    run_paper_trade() 