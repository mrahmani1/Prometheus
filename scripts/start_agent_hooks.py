#!/usr/bin/env python3
"""
Start Intelligent Agent Hook System
Initializes and starts the intelligent agent hook system for automated operations
"""

from pathlib import Path
import asyncio
import logging
import signal
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.system_monitor.intelligent_agent_hooks import IntelligentAgentHookSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "agent_hooks.log"),
    ],
)
logger = logging.getLogger(__name__)


class AgentHookSystemManager:
    """Manager for the intelligent agent hook system"""

    def __init__(self):
        self.hook_system = None
        self.running = False
        self.shutdown_event = asyncio.Event()

    async def start(self):
        """Start the agent hook system"""
        logger.info("Starting Intelligent Agent Hook System")

        try:
            # Initialize the hook system
            self.hook_system = IntelligentAgentHookSystem()

            # Set up signal handlers for graceful shutdown
            self.setup_signal_handlers()

            # Start the event processing
            await self.hook_system.start_event_processing()

            self.running = True
            logger.info("Intelligent Agent Hook System started successfully")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Failed to start agent hook system: {e}")
            raise

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def shutdown(self):
        """Gracefully shutdown the agent hook system"""
        logger.info("Shutting down Intelligent Agent Hook System")

        self.running = False

        if self.hook_system:
            # Wait for active executions to complete (with timeout)
            timeout = 60  # 60 seconds timeout
            start_time = asyncio.get_event_loop().time()

            while (
                self.hook_system.active_executions
                and asyncio.get_event_loop().time() - start_time < timeout
            ):
                logger.info(
                    f"Waiting for {len(self.hook_system.active_executions)} active executions to complete"
                )
                await asyncio.sleep(5)

            if self.hook_system.active_executions:
                logger.warning(
                    f"Timeout reached, {len(self.hook_system.active_executions)} executions still active"
                )

            # Save final execution history
            self.hook_system.save_execution_history()

        self.shutdown_event.set()
        logger.info("Agent hook system shutdown complete")


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Agent Hook System")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Start the system
    manager = AgentHookSystemManager()

    try:
        await manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
