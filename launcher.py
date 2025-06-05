import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict
import threading
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("launcher")

# Load environment variables
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}")


class ServiceManager:
    """Manages the lifecycle of all services."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        self.shutdown_event = threading.Event()

        # Service configurations
        self.services = {
            "mcp_server": {
                "module": "mcp_main",
                "port": int(os.getenv("SERVER_PORT", "8000")),
                "host": os.getenv("SERVER_HOST", "localhost"),
                "description": "MCP Server with integrated services",
            },
            "gradio_interface": {
                "module": "gradio_interface",
                "port": int(os.getenv("GRADIO_PORT", "7860")),
                "host": os.getenv("GRADIO_HOST", "127.0.0.1"),
                "description": "Gradio Web Interface",
            },
        }

        # A2A server runs embedded, not as separate process
        self.a2a_registry = None

    def check_port_available(self, port: int, host: str = "localhost") -> bool:
        """Check if a port is available."""
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception as e:
            logger.warning(f"Error checking port {port}: {e}")
            return False

    def wait_for_service(
        self, port: int, host: str = "localhost", timeout: int = 30
    ) -> bool:
        """Wait for a service to become available on the specified port."""
        import socket

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        return True
            except Exception:
                pass
            time.sleep(1)

        return False

    async def setup_a2a_server(self):
        """Setup the A2A server registry."""
        try:
            from a2a_server import setup_a2a_server

            self.a2a_registry = await setup_a2a_server()
            logger.info("✅ A2A server registry initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup A2A server: {e}")
            return False

    def start_service_process(self, service_name: str, service_config: dict) -> bool:
        """Start a service as a subprocess."""
        try:
            # Check if port is available
            if not self.check_port_available(
                service_config["port"], service_config["host"]
            ):
                logger.error(
                    f"❌ Port {service_config['port']} is already in use for {service_name}"
                )
                return False

            # Prepare environment
            env = os.environ.copy()
            env.update(
                {"PYTHONPATH": str(Path(__file__).parent), "PYTHONUNBUFFERED": "1"}
            )

            # Start the process
            cmd = [sys.executable, "-m", service_config["module"]]
            logger.info(f"🚀 Starting {service_name}: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            self.processes[service_name] = process

            # Start log monitoring thread
            log_thread = threading.Thread(
                target=self._monitor_service_logs,
                args=(service_name, process),
                daemon=True,
            )
            log_thread.start()

            # Wait for service to be ready
            if self.wait_for_service(service_config["port"], service_config["host"]):
                logger.info(
                    f"✅ {service_name} is ready on {service_config['host']}:{service_config['port']}"
                )
                return True
            else:
                logger.error(f"❌ {service_name} failed to start within timeout")
                self.stop_service(service_name)
                return False

        except Exception as e:
            logger.error(f"❌ Error starting {service_name}: {e}")
            return False

    def _monitor_service_logs(self, service_name: str, process: subprocess.Popen):
        """Monitor service logs in a separate thread."""
        try:
            for line in iter(process.stdout.readline, ""):
                if line.strip():
                    logger.info(f"[{service_name}] {line.strip()}")
                if self.shutdown_event.is_set():
                    break
        except Exception as e:
            logger.error(f"Error monitoring logs for {service_name}: {e}")

    def stop_service(self, service_name: str):
        """Stop a specific service."""
        if service_name in self.processes:
            process = self.processes[service_name]
            try:
                logger.info(f"🛑 Stopping {service_name}...")
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {service_name}...")
                    process.kill()
                    process.wait()

                del self.processes[service_name]
                logger.info(f"✅ {service_name} stopped")

            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")

    async def start_all_services(self) -> bool:
        """Start all services in the correct order."""
        logger.info("🚀 Starting A2A-MCP Service Stack...")

        # Step 1: Setup A2A server (embedded)
        logger.info("📡 Setting up A2A server registry...")
        if not await self.setup_a2a_server():
            return False

        # Step 2: Start MCP server
        logger.info("🔧 Starting MCP server...")
        if not self.start_service_process("mcp_server", self.services["mcp_server"]):
            return False

        # Step 3: Wait a bit for MCP server to fully initialize
        await asyncio.sleep(2)

        # Step 4: Start Gradio interface
        logger.info("🌐 Starting Gradio web interface...")
        if not self.start_service_process(
            "gradio_interface", self.services["gradio_interface"]
        ):
            return False

        self.running = True

        # Print success summary
        self._print_startup_summary()

        return True

    def _print_startup_summary(self):
        """Print a summary of running services."""
        logger.info("=" * 60)
        logger.info("🎉 ALL SERVICES STARTED SUCCESSFULLY!")
        logger.info("=" * 60)

        mcp_config = self.services["mcp_server"]
        gradio_config = self.services["gradio_interface"]

        logger.info("📡 A2A Server: Running (embedded)")
        logger.info(f"🔧 MCP Server: http://{mcp_config['host']}:{mcp_config['port']}")
        logger.info(
            f"🌐 Gradio Interface: http://{gradio_config['host']}:{gradio_config['port']}"
        )
        logger.info("")
        logger.info("Available MCP Tools:")
        logger.info("  • get_current_time - Get current MET time")
        logger.info("  • duckduckgo_search - Web search and weather")
        logger.info("  • extract_website_text - Website content extraction")
        logger.info("  • anonymize_text - Text anonymization")
        logger.info("  • convert_to_pdf - File to PDF conversion")
        logger.info("")
        logger.info("Available A2A Agents:")
        logger.info("  • sentiment - Sentiment analysis")
        logger.info("  • optimizer - Text optimization")
        logger.info("  • lektor - Grammar correction")
        logger.info("  • query_ref - Query refactoring")
        logger.info("  • user_interface - Intelligent UI agent")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop all services")
        logger.info("=" * 60)

    def stop_all_services(self):
        """Stop all services."""
        logger.info("🛑 Stopping all services...")
        self.shutdown_event.set()
        self.running = False

        # Stop A2A server
        if self.a2a_registry:
            try:
                asyncio.create_task(self.a2a_registry.stop())
                logger.info("✅ A2A server stopped")
            except Exception as e:
                logger.error(f"Error stopping A2A server: {e}")

        # Stop all service processes
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)

        logger.info("✅ All services stopped")

    def check_services_health(self) -> Dict[str, bool]:
        """Check health of all services."""
        health = {}

        # Check A2A server
        health["a2a_server"] = self.a2a_registry is not None

        # Check service processes
        for service_name, service_config in self.services.items():
            if service_name in self.processes:
                process = self.processes[service_name]
                is_running = process.poll() is None
                port_available = not self.check_port_available(
                    service_config["port"], service_config["host"]
                )
                health[service_name] = is_running and port_available
            else:
                health[service_name] = False

        return health

    async def monitor_services(self):
        """Monitor services and restart if needed."""
        while self.running:
            try:
                health = self.check_services_health()

                for service_name, is_healthy in health.items():
                    if not is_healthy and service_name != "a2a_server":
                        logger.warning(
                            f"⚠️ {service_name} appears unhealthy, attempting restart..."
                        )
                        if service_name in self.services:
                            self.stop_service(service_name)
                            await asyncio.sleep(2)
                            self.start_service_process(
                                service_name, self.services[service_name]
                            )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                await asyncio.sleep(10)


def setup_signal_handlers(service_manager: ServiceManager):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        service_manager.stop_all_services()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main launcher function."""
    logger.info("🚀 A2A-MCP Service Launcher Starting...")

    # Create service manager
    service_manager = ServiceManager()

    # Setup signal handlers
    setup_signal_handlers(service_manager)

    try:
        # Start all services
        if await service_manager.start_all_services():
            # Start service monitoring
            monitor_task = asyncio.create_task(service_manager.monitor_services())

            # Keep running until interrupted
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        else:
            logger.error("❌ Failed to start services")
            return 1

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        service_manager.stop_all_services()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Launcher interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
