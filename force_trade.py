#!/usr/bin/env python3
"""
Force a trade submission for testing Redis logging
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from kellycondor.execution import (
    IBKRClient, 
    LiveIVSkewProcessor, 
    LiveKellySizer, 
    IBKRTradeExecutor,
    run_paper_trade
)
import logging

def force_trade():
    """Force a trade submission for testing."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Forcing a trade submission for testing...")
    
    # Initialize components
    ibkr = IBKRClient()
    
    # Mock connection for testing
    ibkr.connected = True
    ibkr.next_order_id = 1
    
    # Initialize processor and sizer
    strikes = [4400, 4450, 4500, 4550, 4600]
    expiries = ["20241220"]
    proc = LiveIVSkewProcessor(strikes, expiries)
    
    # Set favorable conditions
    proc.current_iv_rank = 0.7  # High IV rank
    proc.current_skew = 0.05    # Low skew
    
    # Historical win rates by IV bucket
    win_rate_table = {
        "low_iv": 0.65,
        "medium_iv": 0.60,
        "high_iv": 0.55
    }
    sizer = LiveKellySizer(proc, win_rate_table)
    
    # Initialize executor
    executor = IBKRTradeExecutor(ibkr, proc, sizer)
    
    # Force submit a trade
    logger.info(f"Current IV Rank: {proc.current_iv_rank}")
    logger.info(f"Current Skew: {proc.current_skew}")
    
    order = executor.submit_iron_condor("SPX", "20241220")
    
    if order:
        logger.info(f"✅ Successfully submitted trade!")
        logger.info(f"   Order ID: {order.order_id}")
        logger.info(f"   Symbol: {order.symbol}")
        logger.info(f"   Call Strike: {order.call_strike}")
        logger.info(f"   Put Strike: {order.put_strike}")
        logger.info(f"   Quantity: {order.quantity}")
        logger.info(f"   Status: {order.status}")
        
        # Check Redis
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        trade_key = f"trade:{order.order_id}"
        
        if r.exists(trade_key):
            trade_data = r.hgetall(trade_key)
            logger.info(f"✅ Trade logged to Redis: {trade_key}")
            for key, value in trade_data.items():
                logger.info(f"   {key.decode()}: {value.decode()}")
        else:
            logger.error("❌ Trade not found in Redis")
    else:
        logger.error("❌ Failed to submit trade")

if __name__ == "__main__":
    force_trade() 