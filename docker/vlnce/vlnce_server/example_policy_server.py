#!/usr/bin/env python3
"""
Example Policy Server for VLN-CE WebSocket Evaluation

This is a simple WebSocket server that demonstrates how to implement
a policy server for the VLN-CE evaluation client. It uses a random
action policy for testing purposes.

Usage:
    python example_policy_server.py --port 8765

To test with the eval client:
    # Terminal 1: Start this server
    python example_policy_server.py --port 8765

    # Terminal 2: Run eval client
    python ws_eval_client.py --server ws://localhost:8765 --split val_seen
"""

import argparse
import asyncio
import logging
from typing import Any, Callable, Dict

import msgpack
import msgpack_numpy
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# Patch msgpack for numpy support
msgpack_numpy.patch()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# Action constants
ACTIONS = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "LOOK_UP",
    5: "LOOK_DOWN",
}


class RandomPolicy:
    """Random action policy matching oracle action distribution."""

    def __init__(self, stop_prob: float = 0.02):
        """
        Args:
            stop_prob: Probability of selecting STOP action
        """
        self.stop_prob = stop_prob
        # Action distribution from VLN-CE paper (excluding STOP)
        # FORWARD: 68%, LEFT: 15%, RIGHT: 15%
        self.action_probs = np.array([
            stop_prob,           # STOP
            0.68 * (1 - stop_prob),  # MOVE_FORWARD
            0.15 * (1 - stop_prob),  # TURN_LEFT
            0.15 * (1 - stop_prob),  # TURN_RIGHT
        ])
        # Normalize
        self.action_probs = self.action_probs / self.action_probs.sum()

    def __call__(self, observation: Dict[str, Any]) -> int:
        """
        Select action based on observation.

        Args:
            observation: Dict with rgb, depth, instruction

        Returns:
            Action integer (0-3 for basic actions)
        """
        return int(np.random.choice(4, p=self.action_probs))


class ForwardPolicy:
    """Simple policy that always moves forward until max steps, then stops."""

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self):
        self.current_step = 0

    def __call__(self, observation: Dict[str, Any]) -> int:
        self.current_step += 1
        if self.current_step >= self.max_steps:
            return 0  # STOP
        return 1  # MOVE_FORWARD


class PolicyServer:
    """WebSocket server that runs a policy for VLN-CE evaluation."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        policy_fn: Callable[[Dict[str, Any]], int] = None
    ):
        """
        Args:
            host: Server host address
            port: Server port
            policy_fn: Function that takes observation dict and returns action int
        """
        self.host = host
        self.port = port
        self.policy_fn = policy_fn or RandomPolicy()
        self.episodes_processed = 0

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a connected evaluation client."""
        logger.info(f"Client connected from {websocket.remote_address}")
        self.episodes_processed = 0

        try:
            async for message in websocket:
                # Deserialize message
                msg = msgpack.unpackb(message, raw=False)
                msg_type = msg.get("type")

                if msg_type == "episode_start":
                    episode_id = msg.get("episode_id", "unknown")
                    instruction = msg.get("instruction", {})
                    logger.info(f"Episode started: {episode_id}")
                    logger.debug(f"Instruction: {instruction.get('text', '')[:50]}...")

                    # Reset policy state if applicable
                    if hasattr(self.policy_fn, "reset"):
                        self.policy_fn.reset()

                elif msg_type == "observation":
                    episode_id = msg.get("episode_id", "unknown")
                    step = msg.get("step", 0)
                    done = msg.get("done", False)

                    if done:
                        # Episode ended, no action needed
                        self.episodes_processed += 1
                        logger.info(f"Episode {episode_id} ended after {step} steps")
                    else:
                        # Get action from policy
                        observation = {
                            "rgb": msg.get("rgb"),
                            "depth": msg.get("depth"),
                            "instruction": msg.get("instruction"),
                        }

                        action = self.policy_fn(observation)

                        # Send action response
                        response = {
                            "type": "action",
                            "action": action,
                        }
                        packed = msgpack.packb(response, use_bin_type=True)
                        await websocket.send(packed)

                        logger.debug(f"Episode {episode_id} step {step}: action={ACTIONS.get(action, action)}")

                elif msg_type == "evaluation_complete":
                    total = msg.get("total_episodes", 0)
                    metrics = msg.get("aggregated_metrics", {})
                    logger.info(f"Evaluation complete: {total} episodes")
                    logger.info(f"Aggregated metrics: {metrics}")

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected. Processed {self.episodes_processed} episodes.")
        except Exception as e:
            logger.exception(f"Error handling client: {e}")

    async def run(self) -> None:
        """Start the WebSocket server."""
        logger.info(f"Starting policy server on {self.host}:{self.port}")
        logger.info(f"Using policy: {type(self.policy_fn).__name__}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=100 * 1024 * 1024,  # 100MB max message size
        ):
            logger.info("Server is ready. Waiting for connections...")
            await asyncio.Future()  # Run forever


def main():
    """Entry point for the example policy server."""
    parser = argparse.ArgumentParser(
        description="Example Policy Server for VLN-CE Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Policies available:
    random  - Random actions matching oracle distribution (default)
    forward - Move forward for N steps then stop

Examples:
    # Start server with random policy
    python example_policy_server.py --port 8765

    # Start server with forward policy
    python example_policy_server.py --port 8765 --policy forward --max-steps 40
        """
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Server port"
    )
    parser.add_argument(
        "--policy", type=str, default="random",
        choices=["random", "forward"],
        help="Policy type to use"
    )
    parser.add_argument(
        "--stop-prob", type=float, default=0.02,
        help="Stop probability for random policy"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps for forward policy"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create policy
    if args.policy == "random":
        policy = RandomPolicy(stop_prob=args.stop_prob)
    elif args.policy == "forward":
        policy = ForwardPolicy(max_steps=args.max_steps)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Run server
    server = PolicyServer(
        host=args.host,
        port=args.port,
        policy_fn=policy,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
