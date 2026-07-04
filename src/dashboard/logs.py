"""
Log management for the Streamlit dashboard.

Provides a thread-safe logging system that can display logs in the Streamlit UI
with automatic scrolling and level filtering.
"""

import logging
import threading
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: datetime
    level: str
    name: str
    message: str
    thread_id: int
    filename: str
    lineno: int
    func_name: str
    traceback: Optional[str] = None


class DashboardLogHandler(logging.Handler):
    """
    Custom logging handler that stores log entries and supports active listeners (e.g., for WebSockets).
    """
    
    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        self.log_entries: List[LogEntry] = []
        self._lock = threading.RLock()
        self._formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self.listeners = []
    
    def add_listener(self, callback) -> None:
        """Register a callback listener that receives new LogEntry objects."""
        with self._lock:
            if callback not in self.listeners:
                self.listeners.append(callback)
                
    def remove_listener(self, callback) -> None:
        """Unregister a callback listener."""
        with self._lock:
            if callback in self.listeners:
                self.listeners.remove(callback)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add a log entry to the internal buffer and notify listeners."""
        try:
            import traceback as tb
            exc_text = None
            if record.exc_info:
                exc_text = "".join(tb.format_exception(*record.exc_info))

            entry = None
            with self._lock:
                entry = LogEntry(
                    timestamp=datetime.now(),
                    level=record.levelname,
                    name=record.name,
                    message=record.getMessage(),
                    thread_id=record.thread,
                    filename=record.filename,
                    lineno=record.lineno,
                    func_name=record.funcName,
                    traceback=exc_text
                )
                
                self.log_entries.append(entry)
                
                if len(self.log_entries) > self.max_entries:
                    self.log_entries = self.log_entries[-self.max_entries:]
                    
                self.periodic_cleanup()
                
            if entry is not None:
                for listener in self.listeners:
                    try:
                        listener(entry)
                    except Exception:
                        pass
                    
        except Exception:
            pass
    
    def get_recent_logs(self, level_filter: str = None, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries with optional level filtering."""
        with self._lock:
            logs = self.log_entries.copy()
            
            if level_filter:
                logs = [log for log in logs if log.level == level_filter]
            
            return logs[-limit:]
    
    def clear(self) -> None:
        """Clear all log entries."""
        with self._lock:
            self.log_entries.clear()
    
    def get_log_count(self) -> int:
        """Get the current number of log entries."""
        with self._lock:
            return len(self.log_entries)
    
    def cleanup_old_logs(self, keep_recent: int = 500) -> None:
        """Clean up old logs, keeping only the most recent entries."""
        with self._lock:
            if len(self.log_entries) > keep_recent:
                self.log_entries = self.log_entries[-keep_recent:]
                return True
        return False
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cleanup to prevent unbounded log growth."""
        if self.get_log_count() % 100 == 0:
            self.cleanup_old_logs(keep_recent=500)


class LogManager:
    """
    Manages logging for the dashboard.
    """
    
    def __init__(self):
        self._handler = DashboardLogHandler()
        self._logger = logging.getLogger("facial-analyzer")
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.INFO)
        
        self._logger.propagate = False
    
    def get_handler(self) -> DashboardLogHandler:
        """Get the log handler."""
        return self._handler
    
    def get_recent_logs(self, level_filter: str = None, limit: int = 100) -> List[LogEntry]:
        """Get recent log entries with optional level filtering."""
        return self._handler.get_recent_logs(level_filter, limit)
    
    def clear_logs(self) -> None:
        """Clear all log entries."""
        self._handler.clear()
    
    def log_system_info(self) -> None:
        """Log system information for debugging."""
        import platform
        import cv2
        
        self._logger.info("System Information:")
        self._logger.info("  Platform: %s", platform.system())
        self._logger.info("  Python: %s", platform.python_version())
        self._logger.info("  OpenCV: %s", cv2.__version__)
        self._logger.info("  MediaPipe: Available")
        
        try:
            import mediapipe as mp
            self._logger.info("  MediaPipe version: %s", mp.__version__)
        except ImportError:
            self._logger.warning("  MediaPipe not available")


_log_manager = LogManager()


def get_log_manager() -> LogManager:
    """Get the global log manager instance."""
    return _log_manager


def setup_dashboard_logging() -> None:
    """Setup logging for the dashboard."""
    log_manager = get_log_manager()
    log_manager.log_system_info()
    log_manager._logger.info("Dashboard logging initialized")