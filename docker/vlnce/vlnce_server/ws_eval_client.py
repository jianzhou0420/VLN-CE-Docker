#!/usr/bin/env python3
"""
WebSocket-based VLN-CE Evaluation Client

This script acts as a WebSocket client that:
1. Connects to an external policy server
2. Sends raw habitat observations to the server
3. Receives actions from the server
4. Steps the habitat environment
5. Collects and saves metrics internally

Usage:
    python ws_eval_client.py --server ws://localhost:8765 --split val_seen
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgpack
import msgpack_numpy
import numpy as np
import websockets
from tqdm import tqdm

import habitat
from habitat import Config, logger as habitat_logger

# Import VLN-CE components
import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config

# Patch msgpack for numpy support
msgpack_numpy.patch()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Protocol version for negotiation
PROTOCOL_VERSION = "1.1"
HANDSHAKE_TIMEOUT = 5.0  # Timeout waiting for server_hello

# Action mapping
ACTION_MAP = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "LOOK_UP",
    5: "LOOK_DOWN",
}
ACTION_NAME_TO_INT = {v: k for k, v in ACTION_MAP.items()}


@dataclass
class ClientConfig:
    """Configuration for the WebSocket evaluation client."""
    server_uri: str = "ws://localhost:8765"
    split: str = "val_seen"
    config_path: str = "vlnce_baselines/config/r2r_baselines/test_set_inference.yaml"
    results_dir: str = "data/eval_results"
    episode_limit: int = -1  # -1 for all episodes
    gpu_id: int = 0
    timeout: float = 300.0  # Action timeout in seconds
    connect_retry_delay: float = 2.0  # Delay between connection retries (seconds)
    panoramic: bool = False  # Enable panoramic observations for waypoint models
    num_panos: int = 12  # Number of panoramic views


class VLNCEEvaluationClient:
    """WebSocket client for VLN-CE evaluation with external policy servers."""

    def __init__(self, client_config: ClientConfig):
        self.client_config = client_config
        self.config: Optional[Config] = None
        self.env: Optional[habitat.Env] = None
        self.websocket = None
        self.stats_episodes: Dict[str, Dict[str, Any]] = {}
        self.current_episode_steps: int = 0
        # Protocol negotiation state
        self.server_capabilities: Optional[Dict[str, Any]] = None
        self._auto_configured: bool = False

    def _setup_config(self) -> Config:
        """Setup habitat configuration for evaluation."""
        config = get_config(self.client_config.config_path)

        # Apply panoramic sensor configuration if needed
        if self.client_config.panoramic:
            from vlnce_baselines.config.default import add_pano_sensors_to_config
            config.defrost()
            config.TASK_CONFIG.TASK.PANO_ROTATIONS = self.client_config.num_panos
            config.freeze()
            config = add_pano_sensors_to_config(config)

        config.defrost()
        # Set evaluation split
        config.TASK_CONFIG.DATASET.SPLIT = self.client_config.split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]

        # Disable shuffling for reproducible evaluation
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1

        # Set GPU
        config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.client_config.gpu_id

        # Ensure required measurements are enabled
        required_measurements = [
            "DISTANCE_TO_GOAL", "SUCCESS", "SPL", "NDTW",
            "PATH_LENGTH", "ORACLE_SUCCESS", "STEPS_TAKEN"
        ]
        for m in required_measurements:
            if m not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append(m)

        # Update NDTW split
        config.TASK_CONFIG.TASK.NDTW.SPLIT = self.client_config.split

        # For waypoint models, ensure GO_TOWARD_POINT action is available
        if self.client_config.panoramic:
            if "GO_TOWARD_POINT" not in config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
                config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "GO_TOWARD_POINT"]

        config.freeze()
        return config

    def _create_env(self) -> habitat.Env:
        """Create a single (non-vectorized) habitat environment."""
        return habitat.Env(config=self.config.TASK_CONFIG)

    def _extract_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format observations for transmission.

        For panoramic mode, stacks multiple RGB/depth views into single arrays.
        """
        if self.client_config.panoramic:
            # Panoramic mode: stack multiple views
            num_panos = self.client_config.num_panos

            # Stack RGB views: [num_panos, H, W, 3]
            rgb_list = [obs.get("rgb", np.zeros((224, 224, 3), dtype=np.uint8))]
            for i in range(1, num_panos):
                key = f"rgb_{i}"
                rgb_i = obs.get(key, np.zeros((224, 224, 3), dtype=np.uint8))
                rgb_list.append(rgb_i)
            rgb = np.stack(rgb_list, axis=0)

            # Stack depth views: [num_panos, H, W, 1]
            depth_list = [obs.get("depth", np.zeros((256, 256, 1), dtype=np.float32))]
            for i in range(1, num_panos):
                key = f"depth_{i}"
                depth_i = obs.get(key, np.zeros((256, 256, 1), dtype=np.float32))
                depth_list.append(depth_i)
            depth = np.stack(depth_list, axis=0)
        else:
            # Egocentric mode: single view
            rgb = obs.get("rgb")
            if rgb is None:
                rgb = np.zeros((256, 256, 3), dtype=np.uint8)

            depth = obs.get("depth")
            if depth is None:
                depth = np.zeros((256, 256, 1), dtype=np.float32)

        # Get instruction data
        instruction_data = obs.get("instruction", {})
        if isinstance(instruction_data, dict):
            instruction = {
                "text": instruction_data.get("text", ""),
                "tokens": instruction_data.get("tokens"),
                "trajectory_id": str(instruction_data.get("trajectory_id", "")),
            }
        else:
            # Fallback for different instruction formats
            instruction = {
                "text": str(instruction_data) if instruction_data else "",
                "tokens": None,
                "trajectory_id": "",
            }

        return {
            "rgb": rgb,
            "depth": depth,
            "instruction": instruction,
        }

    def _validate_action(self, action_data: Any) -> Any:
        """Validate and normalize action from server.

        For discrete actions: returns int (0-5)
        For waypoint actions: returns dict with action and action_args
        """
        if isinstance(action_data, int):
            # Discrete action as integer
            if action_data not in ACTION_MAP:
                raise ValueError(f"Invalid action integer: {action_data}")
            return action_data
        elif isinstance(action_data, dict):
            action_name = action_data.get("action", "")

            # Check for waypoint action format
            if action_name == "GO_TOWARD_POINT":
                action_args = action_data.get("action_args", {})
                if "r" not in action_args or "theta" not in action_args:
                    raise ValueError(f"GO_TOWARD_POINT requires 'r' and 'theta' in action_args")
                # Return waypoint action in format expected by habitat env
                return {
                    "action": "GO_TOWARD_POINT",
                    "action_args": {
                        "r": float(action_args["r"]),
                        "theta": float(action_args["theta"]),
                    }
                }
            elif action_name == "STOP":
                # STOP action can be returned as dict or int
                return 0 if not self.client_config.panoramic else {"action": "STOP"}
            elif action_name in ACTION_NAME_TO_INT:
                # Standard discrete action as string
                return ACTION_NAME_TO_INT[action_name]
            else:
                raise ValueError(f"Invalid action name: {action_name}")
        else:
            raise ValueError(f"Invalid action format: {type(action_data)}")

    async def _send(self, message: Dict) -> None:
        """Serialize and send message via WebSocket."""
        packed = msgpack.packb(message, use_bin_type=True)
        await self.websocket.send(packed)

    async def _receive(self) -> Dict:
        """Receive and deserialize message from WebSocket."""
        try:
            data = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.client_config.timeout
            )
            return msgpack.unpackb(data, raw=False)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout waiting for action (>{self.client_config.timeout}s)")

    async def connect(self) -> None:
        """Establish WebSocket connection and perform protocol negotiation."""
        logger.info(f"Connecting to policy server at {self.client_config.server_uri}")

        while True:
            try:
                self.websocket = await websockets.connect(
                    self.client_config.server_uri,
                    max_size=100 * 1024 * 1024,  # 100MB max message size
                )
                logger.info("Connected to policy server")

                # Perform protocol negotiation
                await self._perform_handshake()
                return
            except (ConnectionRefusedError, OSError) as e:
                logger.info(f"Server not available ({e}), retrying in {self.client_config.connect_retry_delay}s...")
                await asyncio.sleep(self.client_config.connect_retry_delay)

    async def _perform_handshake(self) -> None:
        """Perform protocol negotiation with server.

        Waits for server_hello, auto-configures client, sends client_hello,
        and waits for handshake_complete. Raises error if handshake fails.
        """
        # Wait for server_hello
        try:
            data = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=HANDSHAKE_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Server did not send server_hello within {HANDSHAKE_TIMEOUT}s. "
                "Ensure server is running protocol v1.1."
            )

        msg = msgpack.unpackb(data, raw=False)

        if msg.get("type") != "server_hello":
            raise ConnectionError(
                f"Expected server_hello, got '{msg.get('type')}'. "
                "Server must support protocol v1.1."
            )

        # Process server_hello
        self.server_capabilities = msg.get("capabilities", {})
        protocol_version = msg.get("protocol_version", "1.0")
        server_type = msg.get("server_type", "unknown")

        logger.info(f"Server: type={server_type}, protocol={protocol_version}")
        logger.info(f"Capabilities: mode={self.server_capabilities.get('observation_mode')}, "
                   f"action={self.server_capabilities.get('action_type')}")

        # Auto-configure client based on server capabilities
        self._auto_configure_from_server()

        # Send client_hello
        await self._send_client_hello()

        # Wait for handshake_complete
        try:
            data = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=HANDSHAKE_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Server did not send handshake_complete within {HANDSHAKE_TIMEOUT}s."
            )

        response = msgpack.unpackb(data, raw=False)

        if response.get("type") != "handshake_complete":
            raise ConnectionError(
                f"Expected handshake_complete, got '{response.get('type')}'."
            )

        if response.get("status") == "error":
            error_msg = response.get("message", "Unknown error")
            raise ValueError(f"Handshake failed: {error_msg}")

        logger.info("Handshake complete - client auto-configured")
        self._auto_configured = True

    def _auto_configure_from_server(self) -> None:
        """Auto-configure client based on server capabilities."""
        if not self.server_capabilities:
            return

        obs_mode = self.server_capabilities.get("observation_mode")

        if obs_mode == "panoramic":
            # Enable panoramic mode
            num_panos = self.server_capabilities.get("num_panos", 12)
            if not self.client_config.panoramic:
                logger.info(f"Auto-configuring: panoramic mode with {num_panos} views")
            self.client_config.panoramic = True
            self.client_config.num_panos = num_panos
        elif obs_mode == "egocentric":
            if self.client_config.panoramic:
                logger.info("Auto-configuring: egocentric mode (was panoramic)")
            self.client_config.panoramic = False

    async def _send_client_hello(self) -> None:
        """Send client_hello with current configuration."""
        config = {
            "observation_mode": "panoramic" if self.client_config.panoramic else "egocentric",
            "num_panos": self.client_config.num_panos if self.client_config.panoramic else None,
        }

        # Check compatibility
        server_mode = self.server_capabilities.get("observation_mode", "egocentric")
        compatible = config["observation_mode"] == server_mode

        hello_msg = {
            "type": "client_hello",
            "protocol_version": PROTOCOL_VERSION,
            "client_type": "vlnce_eval",
            "configuration": config,
            "compatible": compatible,
        }

        packed = msgpack.packb(hello_msg, use_bin_type=True)
        await self.websocket.send(packed)
        logger.debug(f"Sent client_hello: mode={config['observation_mode']}, compatible={compatible}")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def run_evaluation(self) -> Dict[str, float]:
        """
        Main evaluation loop - iterates through episodes.

        Returns:
            Aggregated metrics dictionary
        """
        try:
            # Connect to policy server FIRST (includes handshake and auto-config)
            await self.connect()

            # Setup environment AFTER handshake (so auto-config takes effect)
            logger.info("Setting up habitat environment...")
            self.config = self._setup_config()
            self.env = self._create_env()

            total_episodes = len(self.env.episodes)
            if self.client_config.episode_limit > 0:
                total_episodes = min(total_episodes, self.client_config.episode_limit)

            logger.info(f"Starting evaluation: {total_episodes} episodes on {self.client_config.split}")

            # Run evaluation loop
            pbar = tqdm(total=total_episodes, desc="Evaluating")
            episodes_completed = 0
            start_time = time.time()

            while episodes_completed < total_episodes:
                episode_metrics = await self._run_episode()

                if episode_metrics is not None:
                    episodes_completed += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "spl": f"{episode_metrics.get('spl', 0):.3f}",
                        "success": f"{episode_metrics.get('success', 0):.1f}"
                    })

            pbar.close()

            # Aggregate and save results
            elapsed = time.time() - start_time
            logger.info(f"Evaluation complete in {elapsed:.1f}s")

            aggregated = self._aggregate_metrics()
            results_file = self._save_results(aggregated)

            # Notify server of completion
            await self._send({
                "type": "evaluation_complete",
                "total_episodes": episodes_completed,
                "aggregated_metrics": {k: float(v) for k, v in aggregated.items()},
            })

            # Log results
            logger.info(f"Results saved to {results_file}")
            logger.info("Aggregated metrics:")
            for k, v in aggregated.items():
                logger.info(f"  {k}: {v:.4f}")

            return aggregated

        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            raise
        finally:
            await self.disconnect()
            if self.env is not None:
                self.env.close()
                self.env = None

    async def _run_episode(self) -> Optional[Dict[str, Any]]:
        """
        Run a single episode.

        Returns:
            Episode metrics dictionary, or None if episode was skipped
        """
        # Reset environment
        observations = self.env.reset()
        current_episode = self.env.current_episode
        episode_id = current_episode.episode_id

        # Skip if already evaluated
        if episode_id in self.stats_episodes:
            return None

        self.current_episode_steps = 0

        # Extract instruction for episode_start message
        obs_data = self._extract_observations(observations)

        # Send episode start notification
        await self._send({
            "type": "episode_start",
            "episode_id": episode_id,
            "instruction": obs_data["instruction"],
        })

        # Episode loop
        done = False
        while not done:
            # Extract and send observation
            obs_data = self._extract_observations(observations)
            await self._send({
                "type": "observation",
                "episode_id": episode_id,
                "step": self.current_episode_steps,
                "rgb": obs_data["rgb"],
                "depth": obs_data["depth"],
                "instruction": obs_data["instruction"],
                "done": False,
            })

            # Receive action from policy server
            msg = await self._receive()
            if msg.get("type") != "action":
                raise ValueError(f"Expected 'action' message, got: {msg.get('type')}")

            action = self._validate_action(msg.get("action"))

            # Execute action in environment
            observations = self.env.step(action)
            self.current_episode_steps += 1

            # Check if episode is done
            done = self.env.episode_over

        # Send final observation with done=True
        obs_data = self._extract_observations(observations)
        await self._send({
            "type": "observation",
            "episode_id": episode_id,
            "step": self.current_episode_steps,
            "rgb": obs_data["rgb"],
            "depth": obs_data["depth"],
            "instruction": obs_data["instruction"],
            "done": True,
        })

        # Collect episode metrics
        metrics = self.env.get_metrics()
        self.stats_episodes[episode_id] = metrics

        logger.debug(f"Episode {episode_id} complete. SPL: {metrics.get('spl', 0):.4f}")

        return metrics

    def _aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate metrics across all episodes."""
        if not self.stats_episodes:
            return {}

        aggregated = {}
        sample_metrics = next(iter(self.stats_episodes.values()))

        for key in sample_metrics.keys():
            values = [
                ep_metrics[key]
                for ep_metrics in self.stats_episodes.values()
                if isinstance(ep_metrics.get(key), (int, float, np.number))
            ]
            if values:
                aggregated[key] = float(np.mean(values))

        return aggregated

    def _save_results(self, aggregated_metrics: Dict[str, float]) -> Path:
        """Save evaluation results to JSON file."""
        results_dir = Path(self.client_config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"ws_eval_{self.client_config.split}_{timestamp}.json"

        results = {
            "split": self.client_config.split,
            "total_episodes": len(self.stats_episodes),
            "config_path": self.client_config.config_path,
            "server_uri": self.client_config.server_uri,
            "aggregated_metrics": aggregated_metrics,
            "per_episode_metrics": {
                ep_id: {k: float(v) if isinstance(v, (int, float, np.number)) else v
                        for k, v in metrics.items()}
                for ep_id, metrics in self.stats_episodes.items()
            },
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results_file


def main():
    """Entry point for the WebSocket evaluation client."""
    parser = argparse.ArgumentParser(
        description="VLN-CE WebSocket Evaluation Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Connect to local policy server
    python ws_eval_client.py --server ws://localhost:8765 --split val_seen

    # Evaluate on val_unseen with episode limit
    python ws_eval_client.py --server ws://localhost:8765 --split val_unseen --episode-limit 100

    # Use custom config
    python ws_eval_client.py --server ws://localhost:8765 --config path/to/config.yaml
        """
    )
    parser.add_argument(
        "--server", type=str, required=True,
        help="WebSocket URI of the policy server (e.g., ws://localhost:8765)"
    )
    parser.add_argument(
        "--split", type=str, default="val_seen",
        choices=["val_seen", "val_unseen", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--config", type=str,
        default="vlnce_baselines/config/r2r_baselines/test_set_inference.yaml",
        help="Path to habitat config YAML"
    )
    parser.add_argument(
        "--results-dir", type=str, default="data/eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--episode-limit", type=int, default=-1,
        help="Limit number of episodes (-1 for all)"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout in seconds waiting for action from server"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--panoramic", action="store_true",
        help="Enable panoramic observations for waypoint models (HPN/WPN)"
    )
    parser.add_argument(
        "--num-panos", type=int, default=12,
        help="Number of panoramic views (default: 12)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    client_config = ClientConfig(
        server_uri=args.server,
        split=args.split,
        config_path=args.config,
        results_dir=args.results_dir,
        episode_limit=args.episode_limit,
        gpu_id=args.gpu_id,
        timeout=args.timeout,
        panoramic=args.panoramic,
        num_panos=args.num_panos,
    )

    client = VLNCEEvaluationClient(client_config)
    asyncio.run(client.run_evaluation())


if __name__ == "__main__":
    main()
