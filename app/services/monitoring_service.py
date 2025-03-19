import asyncio
import time
from typing import Optional, Callable
from app.logger import setup_logger

logger = setup_logger(__name__)


class ConnectionMonitor:
    """Monitors WebSocket connections and triggers shutdown if inactive."""

    def __init__(
        self,
        connection_manager,
        inactivity_timeout: int = 180,
        stop_callback: Optional[Callable] = None,
    ):
        """Initialize the connection monitor.

        Args:
            connection_manager: The ConnectionManager instance to monitor
            inactivity_timeout: Seconds to wait before triggering shutdown (default: 30)
            stop_callback: Function to call when timeout is reached
        """
        self.connection_manager = connection_manager
        self.inactivity_timeout = inactivity_timeout
        self.stop_callback = stop_callback
        self.timer_task: Optional[asyncio.Task] = None
        self.is_running = False

    def has_active_connections(self) -> bool:
        """Check if there are any active WebSocket connections.

        Returns:
            bool: True if there are active connections, False otherwise
        """
        return len(self.connection_manager.active_connections) > 0

    async def _timer_countdown(self):
        """Run the inactivity timer countdown."""
        try:
            logger.info(
                f"Starting inactivity timer for {self.inactivity_timeout} seconds"
            )
            countdown_start = time.time()

            while time.time() - countdown_start < self.inactivity_timeout:
                # Check if any connections have been established
                if self.has_active_connections():
                    logger.info("Active connection detected, cancelling shutdown timer")
                    return

                # Wait before checking again (1 second intervals)
                print("Time Elapsed:", time.time() - countdown_start)
                await asyncio.sleep(1)

            # Timeout reached with no connections
            logger.warning(
                f"No active connections for {self.inactivity_timeout} seconds, initiating shutdown"
            )
            if self.stop_callback:
                self.stop_callback()
        except asyncio.CancelledError:
            logger.info("Inactivity timer cancelled")
        except Exception as e:
            logger.error(f"Error in inactivity timer: {e}")

    def start_monitoring(self):
        """Start monitoring for WebSocket connections."""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting connection monitor")
        # Start the initial timer
        if not self.has_active_connections():
            self.timer_task = asyncio.create_task(self._timer_countdown())

    def stop_monitoring(self):
        """Stop the connection monitoring."""
        self.is_running = False
        logger.info("Stopping connection monitor")
        if self.timer_task and not self.timer_task.done():
            self.timer_task.cancel()

    async def connection_state_changed(self):
        """Should be called whenever connection state changes."""
        if not self.is_running:
            return

        # If we have active connections, cancel any existing timer
        if self.has_active_connections():
            if self.timer_task and not self.timer_task.done():
                self.timer_task.cancel()
                self.timer_task = None
        # If we have no active connections, start the timer
        elif not self.timer_task or self.timer_task.done():
            self.timer_task = asyncio.create_task(self._timer_countdown())

