"""
Tests for entry window management functionality.
"""

import pytest
from datetime import datetime, time, timedelta
import pytz
from unittest.mock import Mock, patch

from kellycondor.entry_windows import (
    EntryWindow,
    EntryWindowManager,
    WindowAwareProcessor,
    WindowAwareSizer,
    create_default_window_manager,
    create_custom_window_manager
)
from kellycondor.processor import Processor
from kellycondor.sizer import KellySizer


class TestEntryWindow:
    """Test EntryWindow class."""
    
    def test_entry_window_creation(self):
        """Test creating an entry window."""
        window = EntryWindow("Morning", time(9, 30), time(10, 30))
        
        assert window.name == "Morning"
        assert window.start_time == time(9, 30)
        assert window.end_time == time(10, 30)
        assert window.status.value == "inactive"
        assert window.trades_placed == 0
        assert window.total_pnl == 0.0
    
    def test_entry_window_timezone_conversion(self):
        """Test timezone string conversion."""
        window = EntryWindow("Test", time(9, 30), time(10, 30), "US/Eastern")
        
        assert isinstance(window.timezone, pytz.timezone)
        assert window.timezone.zone == "US/Eastern"
    
    def test_is_active(self):
        """Test window active status."""
        window = EntryWindow("Morning", time(9, 30), time(10, 30))
        
        # Test during window
        current_time = datetime.now().replace(hour=9, minute=45)
        assert window.is_active(current_time)
        
        # Test before window
        current_time = datetime.now().replace(hour=9, minute=15)
        assert not window.is_active(current_time)
        
        # Test after window
        current_time = datetime.now().replace(hour=10, minute=45)
        assert not window.is_active(current_time)
    
    def test_is_expired(self):
        """Test window expired status."""
        window = EntryWindow("Morning", time(9, 30), time(10, 30))
        
        # Test before window ends
        current_time = datetime.now().replace(hour=10, minute=15)
        assert not window.is_expired(current_time)
        
        # Test after window ends
        current_time = datetime.now().replace(hour=10, minute=45)
        assert window.is_expired(current_time)
    
    def test_duration_calculation(self):
        """Test window duration calculation."""
        window = EntryWindow("Morning", time(9, 30), time(10, 30))
        
        assert window.get_duration_minutes() == 60


class TestEntryWindowManager:
    """Test EntryWindowManager class."""
    
    def test_default_windows(self):
        """Test default window creation."""
        manager = EntryWindowManager()
        
        assert len(manager.windows) == 4
        window_names = [w.name for w in manager.windows]
        assert "Morning" in window_names
        assert "Mid-Morning" in window_names
        assert "Afternoon" in window_names
        assert "Close" in window_names
    
    def test_custom_windows(self):
        """Test custom window creation."""
        custom_windows = [
            EntryWindow("Custom1", time(10, 0), time(11, 0)),
            EntryWindow("Custom2", time(14, 0), time(15, 0))
        ]
        manager = EntryWindowManager(custom_windows)
        
        assert len(manager.windows) == 2
        assert manager.windows[0].name == "Custom1"
        assert manager.windows[1].name == "Custom2"
    
    def test_get_active_window(self):
        """Test getting active window."""
        manager = EntryWindowManager()
        
        # Mock current time during morning window
        with patch('kellycondor.entry_windows.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now().replace(hour=9, minute=45)
            
            active_window = manager.get_active_window()
            assert active_window is not None
            assert active_window.name == "Morning"
    
    def test_no_active_window(self):
        """Test when no window is active."""
        manager = EntryWindowManager()
        
        # Mock current time outside all windows
        with patch('kellycondor.entry_windows.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now().replace(hour=13, minute=0)
            
            active_window = manager.get_active_window()
            assert active_window is None
    
    def test_record_trade(self):
        """Test recording a trade."""
        manager = EntryWindowManager()
        
        # Record a trade for Morning window
        manager.record_trade("Morning", 100.0, 1000.0)
        
        # Check that the trade was recorded
        morning_window = next(w for w in manager.windows if w.name == "Morning")
        assert morning_window.trades_placed == 1
        assert morning_window.total_pnl == 100.0
        assert morning_window.avg_trade_size == 1000.0
    
    def test_get_window_performance(self):
        """Test getting window performance."""
        manager = EntryWindowManager()
        
        # Record some trades
        manager.record_trade("Morning", 100.0, 1000.0)
        manager.record_trade("Morning", 50.0, 500.0)
        
        performance = manager.get_window_performance("Morning")
        
        assert performance["trades_placed"] == 2
        assert performance["total_pnl"] == 150.0
        assert performance["avg_trade_size"] == 750.0
        assert performance["avg_pnl_per_trade"] == 75.0
    
    def test_get_all_performance(self):
        """Test getting performance for all windows."""
        manager = EntryWindowManager()
        
        # Record trades for multiple windows
        manager.record_trade("Morning", 100.0, 1000.0)
        manager.record_trade("Afternoon", 200.0, 2000.0)
        
        all_performance = manager.get_all_performance()
        
        assert "Morning" in all_performance
        assert "Afternoon" in all_performance
        assert all_performance["Morning"]["trades_placed"] == 1
        assert all_performance["Afternoon"]["trades_placed"] == 1
    
    def test_is_trading_allowed(self):
        """Test trading allowed check."""
        manager = EntryWindowManager()
        
        # Mock current time during window
        with patch('kellycondor.entry_windows.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now().replace(hour=9, minute=45)
            
            assert manager.is_trading_allowed()
        
        # Mock current time outside window
        with patch('kellycondor.entry_windows.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now().replace(hour=13, minute=0)
            
            assert not manager.is_trading_allowed()
    
    def test_get_next_window(self):
        """Test getting next window."""
        manager = EntryWindowManager()
        
        # Mock current time before first window
        with patch('kellycondor.entry_windows.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now().replace(hour=9, minute=0)
            
            next_window = manager.get_next_window()
            assert next_window is not None
            assert next_window.name == "Morning"
    
    def test_get_window_summary(self):
        """Test window summary generation."""
        manager = EntryWindowManager()
        
        # Record some trades
        manager.record_trade("Morning", 100.0, 1000.0)
        
        summary = manager.get_window_summary()
        
        assert "Entry Windows Summary" in summary
        assert "Morning" in summary
        assert "Trades: 1" in summary


class TestWindowAwareProcessor:
    """Test WindowAwareProcessor class."""
    
    def test_window_aware_processor_creation(self):
        """Test creating window-aware processor."""
        base_processor = Mock(spec=Processor)
        window_manager = EntryWindowManager()
        
        processor = WindowAwareProcessor(base_processor, window_manager)
        
        assert processor.base_processor == base_processor
        assert processor.window_manager == window_manager
    
    def test_process_tick_with_window(self):
        """Test processing tick with active window."""
        base_processor = Mock(spec=Processor)
        window_manager = EntryWindowManager()
        processor = WindowAwareProcessor(base_processor, window_manager)
        
        tick_data = {"symbol": "SPX", "iv": 0.25}
        
        # Mock active window
        with patch.object(window_manager, 'get_active_window') as mock_get_window:
            mock_window = Mock()
            mock_window.name = "Morning"
            mock_get_window.return_value = mock_window
            
            processor.process_tick(tick_data)
            
            # Check that base processor was called
            base_processor.process_tick.assert_called_once_with(tick_data)
            
            # Check that window info was added
            assert tick_data["entry_window"] == "Morning"
            assert tick_data["window_active"] is True
    
    def test_process_tick_without_window(self):
        """Test processing tick without active window."""
        base_processor = Mock(spec=Processor)
        window_manager = EntryWindowManager()
        processor = WindowAwareProcessor(base_processor, window_manager)
        
        tick_data = {"symbol": "SPX", "iv": 0.25}
        
        # Mock no active window
        with patch.object(window_manager, 'get_active_window') as mock_get_window:
            mock_get_window.return_value = None
            
            processor.process_tick(tick_data)
            
            # Check that window info was added
            assert tick_data["entry_window"] is None
            assert tick_data["window_active"] is False
    
    def test_should_trade_with_active_window(self):
        """Test should_trade with active window."""
        base_processor = Mock(spec=Processor)
        window_manager = EntryWindowManager()
        processor = WindowAwareProcessor(base_processor, window_manager)
        
        # Mock active window
        with patch.object(window_manager, 'get_active_window') as mock_get_window:
            mock_window = Mock()
            mock_window.name = "Morning"
            mock_get_window.return_value = mock_window
            
            should_trade, window_name = processor.should_trade()
            
            assert should_trade is True
            assert window_name == "Morning"
    
    def test_should_trade_without_active_window(self):
        """Test should_trade without active window."""
        base_processor = Mock(spec=Processor)
        window_manager = EntryWindowManager()
        processor = WindowAwareProcessor(base_processor, window_manager)
        
        # Mock no active window
        with patch.object(window_manager, 'get_active_window') as mock_get_window:
            mock_get_window.return_value = None
            
            should_trade, window_name = processor.should_trade()
            
            assert should_trade is False
            assert window_name is None


class TestWindowAwareSizer:
    """Test WindowAwareSizer class."""
    
    def test_window_aware_sizer_creation(self):
        """Test creating window-aware sizer."""
        base_sizer = Mock(spec=KellySizer)
        window_manager = EntryWindowManager()
        
        sizer = WindowAwareSizer(base_sizer, window_manager)
        
        assert sizer.base_sizer == base_sizer
        assert sizer.window_manager == window_manager
    
    def test_size_position_with_window(self):
        """Test sizing position with window adjustment."""
        base_sizer = Mock(spec=KellySizer)
        window_manager = EntryWindowManager()
        sizer = WindowAwareSizer(base_sizer, window_manager)
        
        # Mock base sizing result
        base_sizing = {"position_size": 1000.0, "max_risk_amount": 100.0}
        base_sizer.size_position.return_value = base_sizing
        
        # Mock window performance
        with patch.object(window_manager, 'get_window_performance') as mock_performance:
            mock_performance.return_value = {"win_rate": 0.6}
            
            result = sizer.size_position(0.5, 0.1, 100000, "Morning")
            
            # Check that base sizer was called
            base_sizer.size_position.assert_called_once_with(0.5, 0.1, 100000)
            
            # Check that adjustment was applied
            assert result["position_size"] == 1000.0  # No adjustment for Morning
            assert result["max_risk_amount"] == 100.0
    
    def test_size_position_with_poor_performance(self):
        """Test sizing position with poor window performance."""
        base_sizer = Mock(spec=KellySizer)
        window_manager = EntryWindowManager()
        sizer = WindowAwareSizer(base_sizer, window_manager)
        
        # Mock base sizing result
        base_sizing = {"position_size": 1000.0, "max_risk_amount": 100.0}
        base_sizer.size_position.return_value = base_sizing
        
        # Mock poor window performance
        with patch.object(window_manager, 'get_window_performance') as mock_performance:
            mock_performance.return_value = {"win_rate": 0.3}
            
            result = sizer.size_position(0.5, 0.1, 100000, "Afternoon")
            
            # Check that adjustment was applied for poor performance
            assert result["position_size"] == 640.0  # 1000 * 0.8 * 0.8
            assert result["max_risk_amount"] == 64.0


class TestWindowManagerFactories:
    """Test window manager factory functions."""
    
    def test_create_default_window_manager(self):
        """Test creating default window manager."""
        manager = create_default_window_manager()
        
        assert len(manager.windows) == 4
        window_names = [w.name for w in manager.windows]
        assert "Morning" in window_names
        assert "Mid-Morning" in window_names
        assert "Afternoon" in window_names
        assert "Close" in window_names
    
    def test_create_custom_window_manager(self):
        """Test creating custom window manager."""
        config = [
            {
                "name": "Custom1",
                "start_time": "10:00",
                "end_time": "11:00",
                "timezone": "US/Eastern"
            },
            {
                "name": "Custom2",
                "start_time": "14:00",
                "end_time": "15:00",
                "timezone": "US/Eastern"
            }
        ]
        
        manager = create_custom_window_manager(config)
        
        assert len(manager.windows) == 2
        assert manager.windows[0].name == "Custom1"
        assert manager.windows[1].name == "Custom2"
        assert manager.windows[0].start_time == time(10, 0)
        assert manager.windows[0].end_time == time(11, 0)


if __name__ == "__main__":
    pytest.main([__file__]) 