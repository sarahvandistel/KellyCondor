"""
Tests for dynamic exit rules functionality.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from kellycondor.exit_rules import (
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


class TestExitTrigger:
    """Test ExitTrigger enum."""
    
    def test_exit_trigger_values(self):
        """Test exit trigger enum values."""
        assert ExitTrigger.TIME_BASED.value == "time_based"
        assert ExitTrigger.IV_CONTRACTION.value == "iv_contraction"
        assert ExitTrigger.THETA_DECAY.value == "theta_decay"
        assert ExitTrigger.TRAILING_PNL.value == "trailing_pnl"
        assert ExitTrigger.MANUAL.value == "manual"


class TestExitReason:
    """Test ExitReason enum."""
    
    def test_exit_reason_values(self):
        """Test exit reason enum values."""
        assert ExitReason.EXPIRY.value == "expiry"
        assert ExitReason.IV_CONTRACTED.value == "iv_contracted"
        assert ExitReason.THETA_THRESHOLD.value == "theta_threshold"
        assert ExitReason.TRAILING_STOP.value == "trailing_stop"
        assert ExitReason.PROFIT_TAKE.value == "profit_take"
        assert ExitReason.LOSS_STOP.value == "loss_stop"
        assert ExitReason.MANUAL.value == "manual"


class TestExitRule:
    """Test ExitRule dataclass."""
    
    def test_exit_rule_creation(self):
        """Test creating an exit rule."""
        rule = ExitRule(
            name="Test Rule",
            trigger=ExitTrigger.TIME_BASED,
            threshold=0.5,
            enabled=True,
            priority=1,
            time_before_expiry=timedelta(hours=2)
        )
        
        assert rule.name == "Test Rule"
        assert rule.trigger == ExitTrigger.TIME_BASED
        assert rule.threshold == 0.5
        assert rule.enabled is True
        assert rule.priority == 1
        assert rule.time_before_expiry == timedelta(hours=2)
    
    def test_exit_rule_with_iv_contraction(self):
        """Test creating an IV contraction exit rule."""
        rule = ExitRule(
            name="IV Contraction",
            trigger=ExitTrigger.IV_CONTRACTION,
            threshold=0.3,
            iv_contraction_threshold=0.3,
            iv_lookback_period=5
        )
        
        assert rule.trigger == ExitTrigger.IV_CONTRACTION
        assert rule.iv_contraction_threshold == 0.3
        assert rule.iv_lookback_period == 5
    
    def test_exit_rule_with_theta_decay(self):
        """Test creating a theta decay exit rule."""
        rule = ExitRule(
            name="Theta Decay",
            trigger=ExitTrigger.THETA_DECAY,
            threshold=0.7,
            theta_decay_threshold=0.7,
            theta_calculation_method="daily"
        )
        
        assert rule.trigger == ExitTrigger.THETA_DECAY
        assert rule.theta_decay_threshold == 0.7
        assert rule.theta_calculation_method == "daily"
    
    def test_exit_rule_with_trailing_pnl(self):
        """Test creating a trailing PnL exit rule."""
        rule = ExitRule(
            name="Trailing PnL",
            trigger=ExitTrigger.TRAILING_PNL,
            threshold=0.0,
            trailing_stop_distance=50,
            trailing_stop_activation=100,
            max_profit_take=200,
            max_loss_stop=-100
        )
        
        assert rule.trigger == ExitTrigger.TRAILING_PNL
        assert rule.trailing_stop_distance == 50
        assert rule.trailing_stop_activation == 100
        assert rule.max_profit_take == 200
        assert rule.max_loss_stop == -100


class TestExitDecision:
    """Test ExitDecision dataclass."""
    
    def test_exit_decision_creation(self):
        """Test creating an exit decision."""
        decision = ExitDecision(
            should_exit=True,
            trigger=ExitTrigger.TIME_BASED,
            reason=ExitReason.EXPIRY,
            threshold_met=2.0,
            current_value=1.5,
            reasoning="Time to expiry: 1.5h, Threshold: 2.0h",
            timestamp=datetime.now()
        )
        
        assert decision.should_exit is True
        assert decision.trigger == ExitTrigger.TIME_BASED
        assert decision.reason == ExitReason.EXPIRY
        assert decision.threshold_met == 2.0
        assert decision.current_value == 1.5
        assert "Time to expiry" in decision.reasoning


class TestExitRuleManager:
    """Test ExitRuleManager class."""
    
    def test_manager_creation(self):
        """Test creating an exit rule manager."""
        manager = ExitRuleManager()
        
        assert manager.rules is not None
        assert len(manager.rules) > 0
        assert manager.exit_history == []
        assert manager.active_positions == {}
    
    def test_default_rules(self):
        """Test default exit rules."""
        manager = ExitRuleManager()
        
        # Check that default rules are created
        rule_names = [rule.name for rule in manager.rules]
        assert "Time-Based Close" in rule_names
        assert "IV Contraction" in rule_names
        assert "Theta Decay" in rule_names
        assert "Trailing PnL Stop" in rule_names
    
    def test_add_position(self):
        """Test adding a position to track."""
        manager = ExitRuleManager()
        
        entry_data = {
            "order_id": 123,
            "symbol": "SPX",
            "entry_iv": 0.25,
            "entry_time": datetime.now(),
            "expiry_time": datetime.now() + timedelta(days=30)
        }
        
        manager.add_position("pos_123", entry_data)
        
        assert "pos_123" in manager.active_positions
        assert manager.active_positions["pos_123"]["entry_data"] == entry_data
    
    def test_remove_position(self):
        """Test removing a position from tracking."""
        manager = ExitRuleManager()
        
        entry_data = {"order_id": 123}
        manager.add_position("pos_123", entry_data)
        
        assert "pos_123" in manager.active_positions
        
        manager.remove_position("pos_123")
        
        assert "pos_123" not in manager.active_positions
    
    def test_evaluate_time_based_exit(self):
        """Test evaluating time-based exit rule."""
        manager = ExitRuleManager()
        
        # Add a position
        entry_data = {
            "expiry_time": datetime.now() + timedelta(hours=1)  # Expires in 1 hour
        }
        manager.add_position("pos_123", entry_data)
        
        # Create a rule that exits 2 hours before expiry
        rule = ExitRule(
            name="Test Time Rule",
            trigger=ExitTrigger.TIME_BASED,
            threshold=0.0,
            time_before_expiry=timedelta(hours=2)
        )
        
        # Should not exit yet (1 hour to expiry, but rule triggers at 2 hours)
        decision = manager._evaluate_time_based(rule, "pos_123", 
                                              manager.active_positions["pos_123"], {})
        
        assert decision.should_exit is False
        
        # Now test with position expiring in 30 minutes
        entry_data["expiry_time"] = datetime.now() + timedelta(minutes=30)
        manager.active_positions["pos_123"]["entry_data"] = entry_data
        
        decision = manager._evaluate_time_based(rule, "pos_123", 
                                              manager.active_positions["pos_123"], {})
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.EXPIRY
    
    def test_evaluate_iv_contraction_exit(self):
        """Test evaluating IV contraction exit rule."""
        manager = ExitRuleManager()
        
        # Add a position
        entry_data = {
            "entry_iv": 0.30  # Entry IV was 30%
        }
        manager.add_position("pos_123", entry_data)
        
        # Create a rule that exits on 30% IV contraction
        rule = ExitRule(
            name="Test IV Rule",
            trigger=ExitTrigger.IV_CONTRACTION,
            threshold=0.3,
            iv_contraction_threshold=0.3
        )
        
        # Test with current IV of 25% (16.7% contraction)
        current_data = {"current_iv": 0.25}
        decision = manager._evaluate_iv_contraction(rule, "pos_123", 
                                                  manager.active_positions["pos_123"], current_data)
        
        assert decision.should_exit is False
        
        # Test with current IV of 20% (33.3% contraction)
        current_data = {"current_iv": 0.20}
        decision = manager._evaluate_iv_contraction(rule, "pos_123", 
                                                  manager.active_positions["pos_123"], current_data)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.IV_CONTRACTED
    
    def test_evaluate_theta_decay_exit(self):
        """Test evaluating theta decay exit rule."""
        manager = ExitRuleManager()
        
        # Add a position
        entry_time = datetime.now() - timedelta(days=5)  # Entered 5 days ago
        expiry_time = datetime.now() + timedelta(days=5)  # Expires in 5 days
        
        entry_data = {
            "entry_time": entry_time,
            "expiry_time": expiry_time
        }
        manager.add_position("pos_123", entry_data)
        manager.active_positions["pos_123"]["entry_time"] = entry_time
        
        # Create a rule that exits at 70% theta decay
        rule = ExitRule(
            name="Test Theta Rule",
            trigger=ExitTrigger.THETA_DECAY,
            threshold=0.7,
            theta_decay_threshold=0.7
        )
        
        # Test with 50% theta decay
        decision = manager._evaluate_theta_decay(rule, "pos_123", 
                                               manager.active_positions["pos_123"], {})
        
        assert decision.should_exit is False
        
        # Test with 80% theta decay (move entry time to 8 days ago)
        entry_time = datetime.now() - timedelta(days=8)
        manager.active_positions["pos_123"]["entry_time"] = entry_time
        manager.active_positions["pos_123"]["entry_data"]["entry_time"] = entry_time
        
        decision = manager._evaluate_theta_decay(rule, "pos_123", 
                                               manager.active_positions["pos_123"], {})
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.THETA_THRESHOLD
    
    def test_evaluate_trailing_pnl_exit(self):
        """Test evaluating trailing PnL exit rule."""
        manager = ExitRuleManager()
        
        # Add a position
        entry_data = {}
        manager.add_position("pos_123", entry_data)
        
        # Create a rule with trailing stop
        rule = ExitRule(
            name="Test Trailing Rule",
            trigger=ExitTrigger.TRAILING_PNL,
            threshold=0.0,
            trailing_stop_distance=50,
            trailing_stop_activation=100,
            max_profit_take=200,
            max_loss_stop=-100
        )
        
        # Test profit take
        current_data = {"current_pnl": 250}  # Above max profit take
        decision = manager._evaluate_trailing_pnl(rule, "pos_123", 
                                                manager.active_positions["pos_123"], current_data)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.PROFIT_TAKE
        
        # Test loss stop
        current_data = {"current_pnl": -150}  # Below max loss stop
        decision = manager._evaluate_trailing_pnl(rule, "pos_123", 
                                                manager.active_positions["pos_123"], current_data)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.LOSS_STOP
        
        # Test trailing stop (first need to activate it)
        manager.active_positions["pos_123"]["max_profit"] = 150  # Above activation
        current_data = {"current_pnl": 80}  # Below trailing stop level (150 - 50 = 100)
        decision = manager._evaluate_trailing_pnl(rule, "pos_123", 
                                                manager.active_positions["pos_123"], current_data)
        
        assert decision.should_exit is True
        assert decision.reason == ExitReason.TRAILING_STOP
    
    def test_evaluate_exits(self):
        """Test evaluating all exit rules."""
        manager = ExitRuleManager()
        
        # Add a position
        entry_data = {
            "expiry_time": datetime.now() + timedelta(minutes=30)  # Expires soon
        }
        manager.add_position("pos_123", entry_data)
        
        # Evaluate exits
        current_data = {"current_pnl": 0.0}
        decisions = manager.evaluate_exits(current_data)
        
        # Should have at least one exit decision (time-based)
        assert len(decisions) > 0
        assert decisions[0].should_exit is True
    
    def test_get_exit_summary(self):
        """Test getting exit summary."""
        manager = ExitRuleManager()
        
        # Add a position with exit decision
        entry_data = {"order_id": 123}
        manager.add_position("pos_123", entry_data)
        
        # Add an exit decision
        decision = ExitDecision(
            should_exit=True,
            trigger=ExitTrigger.TIME_BASED,
            reason=ExitReason.EXPIRY,
            threshold_met=2.0,
            current_value=1.5,
            reasoning="Test",
            timestamp=datetime.now()
        )
        manager.active_positions["pos_123"]["exit_decisions"].append(decision)
        
        summary = manager.get_exit_summary()
        
        assert "Exit Rule Performance" in summary
        assert "expiry" in summary
    
    def test_get_exit_statistics(self):
        """Test getting exit statistics."""
        manager = ExitRuleManager()
        
        # Add a position with exit decision
        entry_data = {"order_id": 123}
        manager.add_position("pos_123", entry_data)
        
        # Add an exit decision
        decision = ExitDecision(
            should_exit=True,
            trigger=ExitTrigger.TIME_BASED,
            reason=ExitReason.EXPIRY,
            threshold_met=2.0,
            current_value=100.0,
            reasoning="Test",
            timestamp=datetime.now()
        )
        manager.active_positions["pos_123"]["exit_decisions"].append(decision)
        
        stats = manager.get_exit_statistics()
        
        assert stats["total_exits"] == 1
        assert "expiry" in stats["exits_by_reason"]
        assert stats["exits_by_reason"]["expiry"] == 1


class TestExitRuleBacktester:
    """Test ExitRuleBacktester class."""
    
    def test_backtester_creation(self):
        """Test creating an exit rule backtester."""
        backtester = ExitRuleBacktester()
        
        assert backtester.base_rules == []
        assert backtester.results == {}
    
    def test_add_exit_configuration(self):
        """Test adding an exit configuration."""
        backtester = ExitRuleBacktester()
        
        rules = [
            ExitRule(
                name="Test Rule",
                trigger=ExitTrigger.TIME_BASED,
                threshold=0.0
            )
        ]
        
        backtester.add_exit_configuration("Test Config", rules)
        
        assert "Test Config" in backtester.results
        assert backtester.results["Test Config"]["rules"] == rules
    
    def test_run_backtest(self):
        """Test running a backtest."""
        backtester = ExitRuleBacktester()
        
        # Add a configuration
        rules = [
            ExitRule(
                name="Time-Based Only",
                trigger=ExitTrigger.TIME_BASED,
                threshold=0.0,
                time_before_expiry=timedelta(hours=2)
            )
        ]
        backtester.add_exit_configuration("Test Config", rules)
        
        # Create mock historical data
        import pandas as pd
        historical_data = pd.DataFrame({
            "price": [4500] * 10,
            "iv": [0.25] * 10,
            "pnl": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "timestamp": [datetime.now() + timedelta(hours=i) for i in range(10)]
        })
        
        # Create entry signals
        entry_signals = [
            {
                "price": 4500,
                "iv": 0.25,
                "timestamp": datetime.now(),
                "expiry_time": datetime.now() + timedelta(days=30)
            }
        ]
        
        # Run backtest
        results = backtester.run_backtest(historical_data, entry_signals)
        
        assert "Test Config" in results
        assert "trades" in results["Test Config"]
        assert "total_pnl" in results["Test Config"]
        assert "win_rate" in results["Test Config"]
    
    def test_compare_configurations(self):
        """Test comparing configurations."""
        backtester = ExitRuleBacktester()
        
        # Add configurations
        rules1 = [ExitRule(name="Rule1", trigger=ExitTrigger.TIME_BASED, threshold=0.0)]
        rules2 = [ExitRule(name="Rule2", trigger=ExitTrigger.IV_CONTRACTION, threshold=0.3)]
        
        backtester.add_exit_configuration("Config1", rules1)
        backtester.add_exit_configuration("Config2", rules2)
        
        # Mock some results
        backtester.results["Config1"]["total_pnl"] = 100.0
        backtester.results["Config1"]["win_rate"] = 0.6
        backtester.results["Config1"]["max_drawdown"] = -50.0
        backtester.results["Config1"]["trades"] = [{"exit_reason": "expiry"}]
        
        backtester.results["Config2"]["total_pnl"] = 150.0
        backtester.results["Config2"]["win_rate"] = 0.7
        backtester.results["Config2"]["max_drawdown"] = -30.0
        backtester.results["Config2"]["trades"] = [{"exit_reason": "iv_contracted"}]
        
        report = backtester.compare_configurations()
        
        assert "Exit Rule Configuration Comparison" in report
        assert "Config1" in report
        assert "Config2" in report
        assert "Total PnL" in report


class TestExitRuleFactories:
    """Test exit rule factory functions."""
    
    def test_create_default_exit_manager(self):
        """Test creating default exit manager."""
        manager = create_default_exit_manager()
        
        assert isinstance(manager, ExitRuleManager)
        assert len(manager.rules) > 0
    
    def test_create_custom_exit_manager(self):
        """Test creating custom exit manager."""
        rules = [
            ExitRule(
                name="Custom Rule",
                trigger=ExitTrigger.TIME_BASED,
                threshold=0.0
            )
        ]
        
        manager = create_custom_exit_manager(rules)
        
        assert isinstance(manager, ExitRuleManager)
        assert manager.rules == rules
    
    def test_create_exit_backtester(self):
        """Test creating exit backtester."""
        backtester = create_exit_backtester()
        
        assert isinstance(backtester, ExitRuleBacktester)


if __name__ == "__main__":
    pytest.main([__file__]) 