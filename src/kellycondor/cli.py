#!/usr/bin/env python3
"""
KellyCondor Live Paper Trading CLI
"""

import argparse
import logging
import sys
import json
from typing import List, Dict, Any
from .execution import run_paper_trade, run_backtest_with_windows


def parse_window_config(config_str: str) -> List[Dict[str, Any]]:
    """Parse window configuration from string or file."""
    try:
        # Try to parse as JSON string
        return json.loads(config_str)
    except json.JSONDecodeError:
        # Try to load from file
        try:
            with open(config_str, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Window config not found: {config_str}")


def get_default_window_config() -> List[Dict[str, Any]]:
    """Get default window configuration."""
    return [
        {
            "name": "Morning",
            "start_time": "09:30",
            "end_time": "10:30",
            "timezone": "US/Eastern"
        },
        {
            "name": "Mid-Morning", 
            "start_time": "11:00",
            "end_time": "12:00",
            "timezone": "US/Eastern"
        },
        {
            "name": "Afternoon",
            "start_time": "14:00", 
            "end_time": "15:00",
            "timezone": "US/Eastern"
        },
        {
            "name": "Close",
            "start_time": "15:30",
            "end_time": "16:00", 
            "timezone": "US/Eastern"
        }
    ]


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
  
  # Paper trade with entry windows enabled
  kelly-live --paper --enable-windows
  
  # Paper trade with custom window configuration
  kelly-live --paper --enable-windows --window-config windows.json
  
  # Backtest with entry window analysis
  kelly-live --backtest --enable-windows --data-file historical_data.csv
        """
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (requires TWS/IB Gateway)"
    )
    
    mode_group.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (no IBKR connection)"
    )
    
    mode_group.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest with historical data"
    )
    
    # IBKR connection arguments
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
    
    # Entry window arguments
    parser.add_argument(
        "--enable-windows",
        action="store_true",
        help="Enable entry window management"
    )
    
    parser.add_argument(
        "--window-config",
        type=str,
        help="Window configuration file or JSON string"
    )
    
    parser.add_argument(
        "--disable-windows",
        action="store_true",
        help="Disable entry window management (default)"
    )
    
    # Backtesting arguments
    parser.add_argument(
        "--data-file",
        type=str,
        help="Historical data file for backtesting (CSV format)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for backtest results"
    )
    
    # General arguments
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
    
    # Parse window configuration
    window_config = None
    if args.window_config:
        try:
            window_config = parse_window_config(args.window_config)
            logger.info(f"Loaded custom window configuration: {len(window_config)} windows")
        except Exception as e:
            logger.error(f"Failed to parse window configuration: {e}")
            sys.exit(1)
    elif args.enable_windows:
        window_config = get_default_window_config()
        logger.info("Using default window configuration")
    
    # Validate arguments
    if args.backtest and not args.data_file:
        logger.error("Backtest mode requires --data-file")
        sys.exit(1)
    
    if args.paper and args.port == 7497:
        logger.info("Using paper trading port 7497")
    elif args.paper and args.port == 7496:
        logger.warning("Using live trading port 7496 - be careful!")
    
    # Display configuration
    logger.info("KellyCondor Live Trading Configuration:")
    if args.paper:
        logger.info("  Mode: Paper Trading")
    elif args.simulate:
        logger.info("  Mode: Simulation")
    elif args.backtest:
        logger.info("  Mode: Backtest")
    
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Client ID: {args.client_id}")
    logger.info(f"  Account Size: ${args.account_size:,.2f}")
    logger.info(f"  Entry Windows: {'Enabled' if args.enable_windows else 'Disabled'}")
    if args.enable_windows and window_config:
        logger.info(f"  Window Config: {len(window_config)} windows")
    logger.info(f"  Dry Run: {args.dry_run}")
    
    try:
        # Start trading based on mode
        if args.paper or args.simulate:
            simulation_mode = args.simulate
            enable_windows = args.enable_windows
            
            if args.paper:
                logger.info("Starting paper trading session...")
                logger.info("Make sure TWS/IB Gateway is running and API connections are enabled")
            else:
                logger.info("Starting simulation mode...")
            
            run_paper_trade(
                api_key=args.api_key,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                simulation_mode=simulation_mode,
                enable_windows=enable_windows,
                window_config=window_config
            )
            
        elif args.backtest:
            logger.info("Starting backtest with entry window analysis...")
            
            # Load historical data
            try:
                import pandas as pd
                data = pd.read_csv(args.data_file)
                logger.info(f"Loaded {len(data)} data points from {args.data_file}")
            except Exception as e:
                logger.error(f"Failed to load data file: {e}")
                sys.exit(1)
            
            # Run backtest
            results = run_backtest_with_windows(
                historical_data=data,
                window_config=window_config,
                account_size=args.account_size
            )
            
            # Display results
            logger.info("Backtest Results:")
            logger.info(f"  Total Trades: {results['total_trades']}")
            logger.info("\nWindow Performance:")
            for window_name, performance in results['window_performance'].items():
                logger.info(f"  {window_name}:")
                logger.info(f"    Trades: {performance['trades_placed']}")
                logger.info(f"    Total PnL: ${performance['total_pnl']:.2f}")
                logger.info(f"    Win Rate: {performance['win_rate']:.1%}")
                logger.info(f"    Avg Trade Size: ${performance['avg_trade_size']:.2f}")
            
            # Save results if output file specified
            if args.output_file:
                try:
                    import json
                    with open(args.output_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    logger.info(f"Results saved to {args.output_file}")
                except Exception as e:
                    logger.error(f"Failed to save results: {e}")
            
        else:
            # This should never happen due to argument validation
            logger.error("Invalid mode specified")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error during trading session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 