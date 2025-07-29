#!/usr/bin/env python3
"""
KellyCondor Live Paper Trading CLI
"""

import argparse
import logging
import sys
import json
from typing import List, Dict, Any
from .execution import run_paper_trade, run_backtest_with_regime_analysis, run_historical_backtest_with_databento


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
  
  # Paper trade with advanced strike selection
  kelly-live --paper --enable-advanced-strikes
  
  # Paper trade with rotating strike selection
  kelly-live --paper --enable-advanced-strikes --rotation-period 3
  
  # Paper trade forcing top-ranked combinations
  kelly-live --paper --enable-advanced-strikes --force-top-ranked
  
  # Paper trade with dynamic exit rules
  kelly-live --paper --enable-exit-rules
  
  # Paper trade with custom exit configuration
  kelly-live --paper --enable-exit-rules --exit-config exit_rules.json
  
  # Paper trade with regime analysis
  kelly-live --paper --enable-regime-analysis
  
  # Paper trade with custom regime clustering
  kelly-live --paper --enable-regime-analysis --regime-clusters 8 --min-trades-per-regime 15
  
  # Combine all features
  kelly-live --paper --enable-windows --enable-advanced-strikes --enable-exit-rules --enable-regime-analysis
  
  # Backtest with regime analysis
  kelly-live --backtest --enable-regime-analysis --data-file historical_data.csv
  
  # Backtest with custom regime configuration
  kelly-live --backtest --enable-regime-analysis --regime-clusters 10 --data-file data.csv
  
  # Backtest with all features
  kelly-live --backtest --enable-windows --enable-advanced-strikes --enable-exit-rules --enable-regime-analysis --data-file data.csv
  
  # Historical backtest with Databento data
  kelly-live --backtest --historical-symbol ES --start-date 2024-01-01 --end-date 2024-01-31 --enable-regime-analysis
  
  # Historical backtest with all features
  kelly-live --backtest --historical-symbol SPX --start-date 2024-01-01 --end-date 2024-01-31 --enable-windows --enable-advanced-strikes --enable-exit-rules --enable-regime-analysis
  
  # Historical backtest with custom dataset
  kelly-live --backtest --historical-symbol ES --start-date 2024-01-01 --end-date 2024-01-31 --dataset GLBX.MDP3 --enable-regime-analysis
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
    
    # Advanced strike selection arguments
    parser.add_argument(
        "--enable-advanced-strikes",
        action="store_true",
        help="Enable advanced strike selection with IV percentile and skew buckets"
    )
    
    parser.add_argument(
        "--rotation-period",
        type=int,
        default=5,
        help="Rotation period for rotating strike selector (default: 5)"
    )
    
    parser.add_argument(
        "--force-top-ranked",
        action="store_true",
        help="Force selection of top-ranked IV/skew combinations only"
    )
    
    parser.add_argument(
        "--strike-config",
        type=str,
        help="Strike selection configuration file"
    )
    
    # Exit rule arguments
    parser.add_argument(
        "--enable-exit-rules",
        action="store_true",
        help="Enable dynamic exit rules (IV contraction, theta decay, trailing PnL)"
    )
    
    parser.add_argument(
        "--exit-config",
        type=str,
        help="Exit rule configuration file"
    )
    
    parser.add_argument(
        "--disable-exit-rules",
        action="store_true",
        help="Disable exit rules (default)"
    )
    
    # Regime analysis arguments
    parser.add_argument(
        "--enable-regime-analysis",
        action="store_true",
        help="Enable regime analysis with clustering by realized volatility and directional drift"
    )
    
    parser.add_argument(
        "--regime-clusters",
        type=int,
        default=6,
        help="Number of regime clusters (default: 6)"
    )
    
    parser.add_argument(
        "--min-trades-per-regime",
        type=int,
        default=10,
        help="Minimum trades per regime cluster (default: 10)"
    )
    
    parser.add_argument(
        "--disable-regime-analysis",
        action="store_true",
        help="Disable regime analysis (default)"
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
    
    # Historical backtesting with Databento
    parser.add_argument(
        "--historical-symbol",
        type=str,
        help="Symbol for historical backtesting (e.g., ES, SPX)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for historical data (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--end-date", 
        type=str,
        help="End date for historical data (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="GLBX.MDP3",
        help="Databento dataset name (default: GLBX.MDP3)"
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
    
    # Parse strike selection configuration
    strike_config = None
    if args.strike_config:
        try:
            with open(args.strike_config, 'r') as f:
                strike_config = json.load(f)
            logger.info(f"Loaded custom strike configuration: {strike_config}")
        except Exception as e:
            logger.error(f"Failed to parse strike configuration: {e}")
            sys.exit(1)
    
    # Parse exit rule configuration
    exit_config = None
    if args.exit_config:
        try:
            with open(args.exit_config, 'r') as f:
                exit_config = json.load(f)
            logger.info(f"Loaded custom exit configuration: {len(exit_config)} rules")
        except Exception as e:
            logger.error(f"Failed to parse exit configuration: {e}")
            sys.exit(1)
    
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
    logger.info(f"  Advanced Strikes: {'Enabled' if args.enable_advanced_strikes else 'Disabled'}")
    if args.enable_advanced_strikes:
        if args.rotation_period > 0:
            logger.info(f"  Rotation Period: {args.rotation_period}")
        if args.force_top_ranked:
            logger.info("  Force Top-Ranked: Enabled")
    logger.info(f"  Exit Rules: {'Enabled' if args.enable_exit_rules else 'Disabled'}")
    if args.enable_exit_rules and exit_config:
        logger.info(f"  Exit Config: {len(exit_config)} rules")
    logger.info(f"  Regime Analysis: {'Enabled' if args.enable_regime_analysis else 'Disabled'}")
    if args.enable_regime_analysis:
        logger.info(f"  Regime Clusters: {args.regime_clusters}")
        logger.info(f"  Min Trades per Regime: {args.min_trades_per_regime}")
    logger.info(f"  Dry Run: {args.dry_run}")
    
    try:
        # Start trading based on mode
        if args.paper or args.simulate:
            simulation_mode = args.simulate
            enable_windows = args.enable_windows
            enable_advanced_strikes = args.enable_advanced_strikes
            rotation_period = args.rotation_period if args.enable_advanced_strikes else 0
            force_top_ranked = args.force_top_ranked
            enable_exit_rules = args.enable_exit_rules
            enable_regime_analysis = args.enable_regime_analysis
            regime_clusters = args.regime_clusters
            min_trades_per_regime = args.min_trades_per_regime
            
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
                window_config=window_config,
                enable_advanced_strikes=enable_advanced_strikes,
                rotation_period=rotation_period,
                force_top_ranked=force_top_ranked,
                enable_exit_rules=enable_exit_rules,
                exit_config=exit_config,
                enable_regime_analysis=enable_regime_analysis,
                regime_clusters=regime_clusters,
                min_trades_per_regime=min_trades_per_regime
            )
            
        elif args.backtest:
            logger.info("Starting backtest with regime analysis...")
            
            # Check if we're doing historical backtesting with Databento
            if args.historical_symbol and args.start_date and args.end_date:
                logger.info("Using Databento historical data for backtesting...")
                
                # Parse dates
                try:
                    from datetime import datetime
                    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
                    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
                except ValueError as e:
                    logger.error(f"Invalid date format. Use YYYY-MM-DD: {e}")
                    sys.exit(1)
                
                # Run historical backtest
                results = run_historical_backtest_with_databento(
                    symbol=args.historical_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    api_key=args.api_key,
                    window_config=window_config,
                    account_size=args.account_size,
                    rotation_period=args.rotation_period if args.enable_advanced_strikes else 0,
                    force_top_ranked=args.force_top_ranked,
                    exit_config=exit_config,
                    regime_clusters=args.regime_clusters,
                    min_trades_per_regime=args.min_trades_per_regime,
                    dataset=args.dataset
                )
                
                # Check for errors
                if "error" in results:
                    logger.error(f"Historical backtest failed: {results['error']}")
                    sys.exit(1)
                
                # Display historical data metadata
                if "historical_data_metadata" in results:
                    metadata = results["historical_data_metadata"]
                    logger.info("\nHistorical Data Metadata:")
                    logger.info(f"  Symbol: {metadata['symbol']}")
                    logger.info(f"  Dataset: {metadata['dataset']}")
                    logger.info(f"  Date Range: {metadata['start_date']} to {metadata['end_date']}")
                    logger.info(f"  Data Points: {metadata['data_points']}")
                
            else:
                # Use local data file
                if not args.data_file:
                    logger.error("For backtesting, either provide --data-file or use --historical-symbol with --start-date and --end-date")
                    sys.exit(1)
                
                # Load historical data
                try:
                    import pandas as pd
                    data = pd.read_csv(args.data_file)
                    logger.info(f"Loaded {len(data)} data points from {args.data_file}")
                except Exception as e:
                    logger.error(f"Failed to load data file: {e}")
                    sys.exit(1)
                
                # Run backtest
                results = run_backtest_with_regime_analysis(
                    historical_data=data,
                    window_config=window_config,
                    account_size=args.account_size,
                    rotation_period=args.rotation_period if args.enable_advanced_strikes else 0,
                    force_top_ranked=args.force_top_ranked,
                    exit_config=exit_config,
                    regime_clusters=args.regime_clusters,
                    min_trades_per_regime=args.min_trades_per_regime
                )
            
            # Display results
            logger.info("Backtest Results:")
            
            # Display regime analysis results
            if "regime_analysis" in results:
                regime_analysis = results["regime_analysis"]
                logger.info("\nRegime Analysis:")
                logger.info(f"  Total Clusters: {regime_analysis['total_clusters']}")
                logger.info(f"  Best Regime: {regime_analysis['best_regime']}")
                
                logger.info("\nRegime Clusters:")
                for cluster_id, cluster_data in regime_analysis["clusters"].items():
                    logger.info(f"  Cluster {cluster_id} ({cluster_data['regime_type']}):")
                    logger.info(f"    Trades: {cluster_data['trade_count']}")
                    logger.info(f"    Win Rate: {cluster_data['win_rate']:.1%}")
                    logger.info(f"    Sharpe Ratio: {cluster_data['sharpe_ratio']:.3f}")
                    logger.info(f"    Avg Reward: ${cluster_data['avg_reward']:.2f}")
                    logger.info(f"    Avg Loss: ${cluster_data['avg_loss']:.2f}")
            
            # Display comparison report
            if "comparison_report" in results:
                logger.info("\n" + results["comparison_report"])
            
            # Display detailed results for each configuration
            for config_name, config_data in results.items():
                if config_name in ["comparison_report", "regime_analysis"]:
                    continue
                    
                logger.info(f"\nConfiguration: {config_name}")
                logger.info(f"  Total PnL: ${config_data['total_pnl']:.2f}")
                logger.info(f"  Win Rate: {config_data['win_rate']:.1%}")
                logger.info(f"  Max Drawdown: ${config_data['max_drawdown']:.2f}")
                logger.info(f"  Total Trades: {len(config_data['trades'])}")
                logger.info(f"  Avg Trade PnL: ${config_data.get('avg_trade_pnl', 0.0):.2f}")
                logger.info(f"  Sharpe Ratio: {config_data.get('sharpe_ratio', 0.0):.3f}")
                logger.info(f"  Max Profit: ${config_data.get('max_profit', 0.0):.2f}")
                logger.info(f"  Max Loss: ${config_data.get('max_loss', 0.0):.2f}")
                
                # Exit reason breakdown
                exit_counts = {}
                for reason in config_data.get("exit_reasons", []):
                    exit_counts[reason] = exit_counts.get(reason, 0) + 1
                
                logger.info("  Exit Reasons:")
                for reason, count in exit_counts.items():
                    logger.info(f"    {reason}: {count}")
            
            # Save results if output file specified
            if args.output_file:
                try:
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