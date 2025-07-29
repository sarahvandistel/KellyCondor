#!/usr/bin/env python3
"""
KellyCondor Live Paper Trading CLI
"""

import argparse
import logging
import sys
from .execution import run_paper_trade


def main():
    """Main CLI entry point for live paper trading."""
    parser = argparse.ArgumentParser(
        description="KellyCondor Live Paper Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trade with default settings
  kelly-live --paper
  
  # Paper trade with custom IBKR settings
  kelly-live --paper --host 127.0.0.1 --port 7496
  
  # Paper trade with custom account size
  kelly-live --paper --account-size 50000
  
  # Simulate trading without IBKR connection
  kelly-live --simulate
        """
    )
    
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (requires TWS/IB Gateway)"
    )
    
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (no IBKR connection)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IBKR TWS/Gateway host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="IBKR TWS/Gateway port (default: 7497 for paper)"
    )
    
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        help="IBKR client ID (default: 1)"
    )
    
    parser.add_argument(
        "--account-size",
        type=float,
        default=100000,
        help="Account size for Kelly sizing (default: 100000)"
    )
    
    parser.add_argument(
        "--api-key",
        default=None,
        help="Databento API key (optional for simulation)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without submitting actual orders"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.paper and not args.simulate:
        logger.error("Must specify either --paper or --simulate mode")
        sys.exit(1)
    
    if args.paper and args.port == 7497:
        logger.info("Using paper trading port 7497")
    elif args.paper and args.port == 7496:
        logger.warning("Using live trading port 7496 - be careful!")
    
    # Display configuration
    logger.info("KellyCondor Live Trading Configuration:")
    logger.info(f"  Mode: {'Paper Trading' if args.paper else 'Simulation'}")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Client ID: {args.client_id}")
    logger.info(f"  Account Size: ${args.account_size:,.2f}")
    logger.info(f"  Dry Run: {args.dry_run}")
    
    try:
        # Start paper trading
        if args.paper:
            logger.info("Starting paper trading session...")
            logger.info("Make sure TWS/IB Gateway is running and API connections are enabled")
            run_paper_trade(
                api_key=args.api_key,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                simulation_mode=False
            )
        elif args.simulate:
            logger.info("Starting simulation mode...")
            run_paper_trade(
                api_key=args.api_key,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                simulation_mode=True
            )
        else:
            # This should never happen due to argument validation
            logger.error("Must specify either --paper or --simulate mode")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error during trading session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 