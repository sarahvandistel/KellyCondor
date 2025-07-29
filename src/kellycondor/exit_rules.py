"""
Dynamic exit rules for KellyCondor iron condor strategy.
Supports IV contraction, theta decay, and trailing PnL stop exits.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class ExitTrigger(Enum):
    """Types of exit triggers."""
    TIME_BASED = "time_based"
    IV_CONTRACTION = "iv_contraction"
    THETA_DECAY = "theta_decay"
    TRAILING_PNL = "trailing_pnl"
    MANUAL = "manual"


class ExitReason(Enum):
    """Exit reasons for tracking."""
    EXPIRY = "expiry"
    IV_CONTRACTED = "iv_contracted"
    THETA_THRESHOLD = "theta_threshold"
    TRAILING_STOP = "trailing_stop"
    PROFIT_TAKE = "profit_take"
    LOSS_STOP = "loss_stop"
    MANUAL = "manual"


@dataclass
class ExitRule:
    """Configuration for an exit rule."""
    name: str
    trigger: ExitTrigger
    threshold: float
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    
    # Time-based specific
    time_before_expiry: Optional[timedelta] = None
    
    # IV contraction specific
    iv_contraction_threshold: Optional[float] = None
    iv_lookback_period: Optional[int] = None  # Days
    
    # Theta decay specific
    theta_decay_threshold: Optional[float] = None  # Percentage
    theta_calculation_method: str = "daily"  # daily, hourly
    
    # Trailing PnL specific
    trailing_stop_distance: Optional[float] = None  # Points
    trailing_stop_activation: Optional[float] = None  # Profit level to activate
    max_profit_take: Optional[float] = None  # Maximum profit to take
    max_loss_stop: Optional[float] = None  # Maximum loss to stop


@dataclass
class ExitDecision:
    """Result of exit rule evaluation."""
    should_exit: bool
    trigger: ExitTrigger
    reason: ExitReason
    threshold_met: float
    current_value: float
    reasoning: str
    timestamp: datetime


class ExitRuleManager:
    """Manages multiple exit rules and evaluates exit conditions."""
    
    def __init__(self, rules: List[ExitRule] = None):
        self.rules = rules or self._get_default_rules()
        self.exit_history = []
        self.active_positions = {}
        
    def _get_default_rules(self) -> List[ExitRule]:
        """Get default exit rules."""
        return [
            ExitRule(
                name="Time-Based Close",
                trigger=ExitTrigger.TIME_BASED,
                threshold=0.0,
                time_before_expiry=timedelta(hours=2),
                priority=1
            ),
            ExitRule(
                name="IV Contraction",
                trigger=ExitTrigger.IV_CONTRACTION,
                threshold=0.3,  # 30% IV contraction
                iv_contraction_threshold=0.3,
                iv_lookback_period=5,
                priority=2
            ),
            ExitRule(
                name="Theta Decay",
                trigger=ExitTrigger.THETA_DECAY,
                threshold=0.7,  # 70% theta decay
                theta_decay_threshold=0.7,
                theta_calculation_method="daily",
                priority=3
            ),
            ExitRule(
                name="Trailing PnL Stop",
                trigger=ExitTrigger.TRAILING_PNL,
                threshold=0.0,
                trailing_stop_distance=50,  # 50 points
                trailing_stop_activation=100,  # Activate at $100 profit
                max_profit_take=200,  # Take profit at $200
                max_loss_stop=-100,  # Stop loss at -$100
                priority=4
            )
        ]
    
    def add_position(self, position_id: str, entry_data: Dict[str, Any]):
        """Add a position to track for exits."""
        self.active_positions[position_id] = {
            "entry_data": entry_data,
            "entry_time": datetime.now(),
            "max_profit": 0.0,
            "current_pnl": 0.0,
            "exit_decisions": []
        }
        logging.info(f"Added position {position_id} for exit tracking")
    
    def remove_position(self, position_id: str):
        """Remove a position from tracking."""
        if position_id in self.active_positions:
            del self.active_positions[position_id]
            logging.info(f"Removed position {position_id} from exit tracking")
    
    def evaluate_exits(self, current_data: Dict[str, Any]) -> List[ExitDecision]:
        """Evaluate all exit rules for all active positions."""
        decisions = []
        
        for position_id, position_data in self.active_positions.items():
            for rule in sorted(self.rules, key=lambda r: r.priority):
                if not rule.enabled:
                    continue
                
                decision = self._evaluate_rule(rule, position_id, position_data, current_data)
                if decision.should_exit:
                    decisions.append(decision)
                    position_data["exit_decisions"].append(decision)
                    break  # Exit on first triggered rule (priority order)
        
        return decisions
    
    def _evaluate_rule(self, rule: ExitRule, position_id: str, 
                      position_data: Dict[str, Any], current_data: Dict[str, Any]) -> ExitDecision:
        """Evaluate a specific exit rule."""
        
        if rule.trigger == ExitTrigger.TIME_BASED:
            return self._evaluate_time_based(rule, position_id, position_data, current_data)
        elif rule.trigger == ExitTrigger.IV_CONTRACTION:
            return self._evaluate_iv_contraction(rule, position_id, position_data, current_data)
        elif rule.trigger == ExitTrigger.THETA_DECAY:
            return self._evaluate_theta_decay(rule, position_id, position_data, current_data)
        elif rule.trigger == ExitTrigger.TRAILING_PNL:
            return self._evaluate_trailing_pnl(rule, position_id, position_data, current_data)
        else:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="Unknown trigger type",
                timestamp=datetime.now()
            )
    
    def _evaluate_time_based(self, rule: ExitRule, position_id: str,
                            position_data: Dict[str, Any], current_data: Dict[str, Any]) -> ExitDecision:
        """Evaluate time-based exit rule."""
        if rule.time_before_expiry is None:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="No time threshold configured",
                timestamp=datetime.now()
            )
        
        # Get expiry time from position data
        expiry_time = position_data["entry_data"].get("expiry_time")
        if not expiry_time:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="No expiry time available",
                timestamp=datetime.now()
            )
        
        current_time = datetime.now()
        time_to_expiry = expiry_time - current_time
        
        should_exit = time_to_expiry <= rule.time_before_expiry
        
        return ExitDecision(
            should_exit=should_exit,
            trigger=rule.trigger,
            reason=ExitReason.EXPIRY if should_exit else ExitReason.MANUAL,
            threshold_met=rule.time_before_expiry.total_seconds() / 3600,  # Hours
            current_value=time_to_expiry.total_seconds() / 3600,
            reasoning=f"Time to expiry: {time_to_expiry.total_seconds()/3600:.1f}h, "
                     f"Threshold: {rule.time_before_expiry.total_seconds()/3600:.1f}h",
            timestamp=current_time
        )
    
    def _evaluate_iv_contraction(self, rule: ExitRule, position_id: str,
                                position_data: Dict[str, Any], current_data: Dict[str, Any]) -> ExitDecision:
        """Evaluate IV contraction exit rule."""
        if rule.iv_contraction_threshold is None:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="No IV contraction threshold configured",
                timestamp=datetime.now()
            )
        
        # Get historical IV data
        historical_iv = current_data.get("historical_iv", [])
        current_iv = current_data.get("current_iv", 0.0)
        
        if not historical_iv:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="No historical IV data available",
                timestamp=datetime.now()
            )
        
        # Calculate IV contraction
        entry_iv = position_data["entry_data"].get("entry_iv", current_iv)
        iv_contraction = (entry_iv - current_iv) / entry_iv if entry_iv > 0 else 0.0
        
        should_exit = iv_contraction >= rule.iv_contraction_threshold
        
        return ExitDecision(
            should_exit=should_exit,
            trigger=rule.trigger,
            reason=ExitReason.IV_CONTRACTED if should_exit else ExitReason.MANUAL,
            threshold_met=rule.iv_contraction_threshold,
            current_value=iv_contraction,
            reasoning=f"IV contraction: {iv_contraction:.1%}, "
                     f"Threshold: {rule.iv_contraction_threshold:.1%}",
            timestamp=datetime.now()
        )
    
    def _evaluate_theta_decay(self, rule: ExitRule, position_id: str,
                             position_data: Dict[str, Any], current_data: Dict[str, Any]) -> ExitDecision:
        """Evaluate theta decay exit rule."""
        if rule.theta_decay_threshold is None:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="No theta decay threshold configured",
                timestamp=datetime.now()
            )
        
        # Calculate theta decay based on time passed
        entry_time = position_data["entry_time"]
        current_time = datetime.now()
        time_passed = current_time - entry_time
        
        # Get position expiry
        expiry_time = position_data["entry_data"].get("expiry_time")
        if not expiry_time:
            return ExitDecision(
                should_exit=False,
                trigger=rule.trigger,
                reason=ExitReason.MANUAL,
                threshold_met=0.0,
                current_value=0.0,
                reasoning="No expiry time available",
                timestamp=datetime.now()
            )
        
        total_time = expiry_time - entry_time
        theta_decay = time_passed / total_time if total_time.total_seconds() > 0 else 0.0
        
        should_exit = theta_decay >= rule.theta_decay_threshold
        
        return ExitDecision(
            should_exit=should_exit,
            trigger=rule.trigger,
            reason=ExitReason.THETA_THRESHOLD if should_exit else ExitReason.MANUAL,
            threshold_met=rule.theta_decay_threshold,
            current_value=theta_decay,
            reasoning=f"Theta decay: {theta_decay:.1%}, "
                     f"Threshold: {rule.theta_decay_threshold:.1%}",
            timestamp=current_time
        )
    
    def _evaluate_trailing_pnl(self, rule: ExitRule, position_id: str,
                              position_data: Dict[str, Any], current_data: Dict[str, Any]) -> ExitDecision:
        """Evaluate trailing PnL exit rule."""
        current_pnl = current_data.get("current_pnl", 0.0)
        position_data["current_pnl"] = current_pnl
        
        # Update max profit
        if current_pnl > position_data["max_profit"]:
            position_data["max_profit"] = current_pnl
        
        should_exit = False
        reason = ExitReason.MANUAL
        reasoning_parts = []
        
        # Check profit take
        if rule.max_profit_take and current_pnl >= rule.max_profit_take:
            should_exit = True
            reason = ExitReason.PROFIT_TAKE
            reasoning_parts.append(f"Profit take: ${current_pnl:.2f}")
        
        # Check loss stop
        elif rule.max_loss_stop and current_pnl <= rule.max_loss_stop:
            should_exit = True
            reason = ExitReason.LOSS_STOP
            reasoning_parts.append(f"Loss stop: ${current_pnl:.2f}")
        
        # Check trailing stop
        elif (rule.trailing_stop_distance and rule.trailing_stop_activation and
              position_data["max_profit"] >= rule.trailing_stop_activation):
            
            trailing_stop_level = position_data["max_profit"] - rule.trailing_stop_distance
            if current_pnl <= trailing_stop_level:
                should_exit = True
                reason = ExitReason.TRAILING_STOP
                reasoning_parts.append(f"Trailing stop: ${current_pnl:.2f} <= ${trailing_stop_level:.2f}")
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No exit conditions met"
        
        return ExitDecision(
            should_exit=should_exit,
            trigger=rule.trigger,
            reason=reason,
            threshold_met=rule.max_profit_take or rule.max_loss_stop or rule.trailing_stop_distance or 0.0,
            current_value=current_pnl,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def get_exit_summary(self) -> str:
        """Get a summary of exit rule performance."""
        summary = "Exit Rule Performance:\n"
        summary += "=" * 50 + "\n"
        
        # Count exits by reason
        exit_counts = {}
        for position_data in self.active_positions.values():
            for decision in position_data["exit_decisions"]:
                reason = decision.reason.value
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
        
        for reason, count in exit_counts.items():
            summary += f"{reason}: {count} exits\n"
        
        summary += f"\nActive Positions: {len(self.active_positions)}"
        
        return summary
    
    def get_exit_statistics(self) -> Dict[str, Any]:
        """Get detailed exit statistics."""
        stats = {
            "total_exits": 0,
            "exits_by_reason": {},
            "exits_by_trigger": {},
            "avg_pnl_by_exit": {},
            "active_positions": len(self.active_positions)
        }
        
        for position_data in self.active_positions.values():
            for decision in position_data["exit_decisions"]:
                stats["total_exits"] += 1
                
                # Count by reason
                reason = decision.reason.value
                stats["exits_by_reason"][reason] = stats["exits_by_reason"].get(reason, 0) + 1
                
                # Count by trigger
                trigger = decision.trigger.value
                stats["exits_by_trigger"][trigger] = stats["exits_by_trigger"].get(trigger, 0) + 1
                
                # Average PnL by exit reason
                if reason not in stats["avg_pnl_by_exit"]:
                    stats["avg_pnl_by_exit"][reason] = []
                stats["avg_pnl_by_exit"][reason].append(decision.current_value)
        
        # Calculate averages
        for reason, pnls in stats["avg_pnl_by_exit"].items():
            stats["avg_pnl_by_exit"][reason] = np.mean(pnls) if pnls else 0.0
        
        return stats


class ExitRuleBacktester:
    """Backtester for comparing different exit rule configurations."""
    
    def __init__(self, base_rules: List[ExitRule] = None):
        self.base_rules = base_rules or []
        self.results = {}
        
    def add_exit_configuration(self, name: str, rules: List[ExitRule]):
        """Add an exit configuration to test."""
        self.results[name] = {
            "rules": rules,
            "trades": [],
            "pnl_curve": [],
            "exit_reasons": [],
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0
        }
    
    def run_backtest(self, historical_data: pd.DataFrame, 
                    entry_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run backtest comparing different exit configurations."""
        
        for config_name, config_data in self.results.items():
            logging.info(f"Running backtest for configuration: {config_name}")
            
            # Initialize exit rule manager
            exit_manager = ExitRuleManager(config_data["rules"])
            
            # Simulate trading with this configuration
            trades = []
            pnl_curve = []
            current_pnl = 0.0
            max_pnl = 0.0
            max_drawdown = 0.0
            
            for entry_signal in entry_signals:
                # Simulate entry
                position_id = f"pos_{len(trades)}"
                entry_data = {
                    "entry_price": entry_signal.get("price", 4500),
                    "entry_iv": entry_signal.get("iv", 0.25),
                    "entry_time": entry_signal.get("timestamp", datetime.now()),
                    "expiry_time": entry_signal.get("expiry_time", datetime.now() + timedelta(days=30))
                }
                
                exit_manager.add_position(position_id, entry_data)
                
                # Simulate holding period
                for _, row in historical_data.iterrows():
                    current_data = {
                        "current_iv": row.get("iv", 0.25),
                        "current_pnl": row.get("pnl", 0.0),
                        "timestamp": row.get("timestamp", datetime.now())
                    }
                    
                    # Check for exits
                    exit_decisions = exit_manager.evaluate_exits(current_data)
                    
                    if exit_decisions:
                        # Position exited
                        decision = exit_decisions[0]  # Take first exit decision
                        exit_pnl = current_data["current_pnl"]
                        
                        trade = {
                            "position_id": position_id,
                            "entry_time": entry_data["entry_time"],
                            "exit_time": decision.timestamp,
                            "entry_price": entry_data["entry_price"],
                            "exit_price": row.get("price", 4500),
                            "pnl": exit_pnl,
                            "exit_reason": decision.reason.value,
                            "exit_trigger": decision.trigger.value,
                            "holding_period": (decision.timestamp - entry_data["entry_time"]).total_seconds() / 3600
                        }
                        
                        trades.append(trade)
                        current_pnl += exit_pnl
                        max_pnl = max(max_pnl, current_pnl)
                        max_drawdown = min(max_drawdown, current_pnl - max_pnl)
                        
                        pnl_curve.append({
                            "timestamp": decision.timestamp,
                            "cumulative_pnl": current_pnl,
                            "trade_pnl": exit_pnl
                        })
                        
                        exit_manager.remove_position(position_id)
                        break
                
                # If position didn't exit, close at expiry
                if position_id in exit_manager.active_positions:
                    final_pnl = historical_data.iloc[-1].get("pnl", 0.0)
                    current_pnl += final_pnl
                    
                    trade = {
                        "position_id": position_id,
                        "entry_time": entry_data["entry_time"],
                        "exit_time": historical_data.iloc[-1].get("timestamp", datetime.now()),
                        "entry_price": entry_data["entry_price"],
                        "exit_price": historical_data.iloc[-1].get("price", 4500),
                        "pnl": final_pnl,
                        "exit_reason": "expiry",
                        "exit_trigger": "time_based",
                        "holding_period": (historical_data.iloc[-1].get("timestamp", datetime.now()) - entry_data["entry_time"]).total_seconds() / 3600
                    }
                    
                    trades.append(trade)
                    exit_manager.remove_position(position_id)
            
            # Calculate statistics
            wins = sum(1 for trade in trades if trade["pnl"] > 0)
            win_rate = wins / len(trades) if trades else 0.0
            
            config_data["trades"] = trades
            config_data["pnl_curve"] = pnl_curve
            config_data["total_pnl"] = current_pnl
            config_data["win_rate"] = win_rate
            config_data["max_drawdown"] = max_drawdown
            config_data["exit_reasons"] = [trade["exit_reason"] for trade in trades]
        
        return self.results
    
    def compare_configurations(self) -> str:
        """Generate comparison report for all configurations."""
        report = "Exit Rule Configuration Comparison:\n"
        report += "=" * 60 + "\n\n"
        
        for config_name, config_data in self.results.items():
            report += f"Configuration: {config_name}\n"
            report += "-" * 40 + "\n"
            report += f"Total PnL: ${config_data['total_pnl']:.2f}\n"
            report += f"Win Rate: {config_data['win_rate']:.1%}\n"
            report += f"Max Drawdown: ${config_data['max_drawdown']:.2f}\n"
            report += f"Total Trades: {len(config_data['trades'])}\n"
            
            # Exit reason breakdown
            exit_counts = {}
            for reason in config_data["exit_reasons"]:
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
            
            report += "Exit Reasons:\n"
            for reason, count in exit_counts.items():
                report += f"  {reason}: {count}\n"
            
            report += "\n"
        
        return report


def create_default_exit_manager() -> ExitRuleManager:
    """Create a default exit rule manager."""
    return ExitRuleManager()


def create_custom_exit_manager(rules: List[ExitRule]) -> ExitRuleManager:
    """Create a custom exit rule manager."""
    return ExitRuleManager(rules)


def create_exit_backtester() -> ExitRuleBacktester:
    """Create an exit rule backtester."""
    return ExitRuleBacktester() 