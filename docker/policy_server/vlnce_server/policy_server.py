"""
WebSocket policy server for VLN-CE evaluation.

Handles WebSocket communication with the eval client.
Delegates all model-specific logic to a PolicyAdapter.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import msgpack
import msgpack_numpy
import websockets
from websockets.server import WebSocketServerProtocol

from vlnce_server.base_adapter import PolicyAdapter

# Patch msgpack for numpy support
msgpack_numpy.patch()

logger = logging.getLogger(__name__)

# Protocol version for negotiation
PROTOCOL_VERSION = "1.1"


class PolicyServer:
    """WebSocket server that delegates to a PolicyAdapter.

    This class handles all WebSocket communication:
    - Receiving observations from the eval client
    - Parsing message types
    - Calling the appropriate adapter methods
    - Sending action responses

    The server is model-agnostic - all model-specific logic
    is handled by the adapter.
    """

    def __init__(
        self,
        adapter: PolicyAdapter,
        host: str = "0.0.0.0",
        port: int = 8765,
    ):
        """Initialize the policy server.

        Args:
            adapter: PolicyAdapter instance for handling observations
            host: Server host address
            port: Server port
        """
        self.adapter = adapter
        self.host = host
        self.port = port
        self.episodes_processed = 0
        self._current_episode_id: Optional[str] = None
        self._handshake_complete: bool = False

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a connected evaluation client.

        Performs protocol negotiation handshake, then processes messages.
        Message types:
        - server_hello: Sent by server on connection (capabilities)
        - client_hello: Client acknowledges and confirms configuration
        - handshake_complete: Server confirms handshake success/failure
        - episode_start: Initialize for new episode
        - observation: Process observation, return action (if not done)
        - evaluation_complete: Log final metrics

        Args:
            websocket: WebSocket connection to client
        """
        logger.info(f"Client connected from {websocket.remote_address}")
        self.episodes_processed = 0
        self._handshake_complete = False

        try:
            # Send server_hello immediately on connection
            await self._send_server_hello(websocket)

            async for message in websocket:
                # Deserialize message
                msg = msgpack.unpackb(message, raw=False)
                msg_type = msg.get("type")

                # Handle handshake messages
                if msg_type == "client_hello":
                    await self._handle_client_hello(msg, websocket)
                    continue

                # Require handshake before processing other messages
                if not self._handshake_complete:
                    logger.error(
                        f"Received '{msg_type}' before handshake complete. "
                        "Client must send client_hello first."
                    )
                    raise ValueError("Handshake required before sending messages")

                if msg_type == "episode_start":
                    await self._handle_episode_start(msg)

                elif msg_type == "observation":
                    await self._handle_observation(msg, websocket)

                elif msg_type == "evaluation_complete":
                    self._handle_evaluation_complete(msg)

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(
                f"Client disconnected. Processed {self.episodes_processed} episodes."
            )
        except Exception as e:
            logger.exception(f"Error handling client: {e}")

    async def _handle_episode_start(self, msg: dict) -> None:
        """Handle episode_start message.

        Args:
            msg: Parsed message dict
        """
        episode_id = msg.get("episode_id", "unknown")
        instruction = msg.get("instruction", {})

        self._current_episode_id = episode_id
        logger.info(f"Episode started: {episode_id}")
        logger.debug(f"Instruction: {instruction.get('text', '')[:50]}...")

        # Delegate to adapter
        self.adapter.on_episode_start(episode_id, instruction)

    async def _handle_observation(
        self, msg: dict, websocket: WebSocketServerProtocol
    ) -> None:
        """Handle observation message.

        If done=False, get action from adapter and send response.
        If done=True, notify adapter that episode ended.

        Args:
            msg: Parsed message dict
            websocket: WebSocket connection for sending response
        """
        episode_id = msg.get("episode_id", "unknown")
        step = msg.get("step", 0)
        done = msg.get("done", False)

        if done:
            # Episode ended
            self.episodes_processed += 1
            logger.info(f"Episode {episode_id} ended after {step} steps")
            self.adapter.on_episode_end()
        else:
            # Get action from adapter
            rgb = msg.get("rgb")
            depth = msg.get("depth")
            instruction = msg.get("instruction", {})

            action = self.adapter.on_observation(rgb, depth, instruction)

            # Send action response
            response = {"type": "action", "action": action}
            packed = msgpack.packb(response, use_bin_type=True)
            await websocket.send(packed)

            logger.debug(f"Episode {episode_id} step {step}: action={action}")

    def _handle_evaluation_complete(self, msg: dict) -> None:
        """Handle evaluation_complete message.

        Args:
            msg: Parsed message dict
        """
        total = msg.get("total_episodes", 0)
        metrics = msg.get("aggregated_metrics", {})

        logger.info(f"Evaluation complete: {total} episodes")
        logger.info(f"Aggregated metrics: {metrics}")

    # === Protocol Negotiation (v1.1) ===

    async def _send_server_hello(self, websocket: WebSocketServerProtocol) -> None:
        """Send server_hello message with capabilities.

        Args:
            websocket: WebSocket connection
        """
        capabilities = self.adapter.get_capabilities()

        hello_msg: Dict[str, Any] = {
            "type": "server_hello",
            "protocol_version": PROTOCOL_VERSION,
            "server_type": self.adapter.server_type,
            "capabilities": capabilities,
        }

        packed = msgpack.packb(hello_msg, use_bin_type=True)
        await websocket.send(packed)

        logger.info(
            f"Sent server_hello (type={self.adapter.server_type}, "
            f"mode={capabilities['observation_mode']})"
        )

    async def _handle_client_hello(
        self, msg: dict, websocket: WebSocketServerProtocol
    ) -> None:
        """Handle client_hello message and complete handshake.

        Args:
            msg: Parsed client_hello message
            websocket: WebSocket connection
        """
        client_version = msg.get("protocol_version", "1.0")
        client_config = msg.get("configuration", {})
        compatible = msg.get("compatible", True)

        logger.info(
            f"Received client_hello (version={client_version}, compatible={compatible})"
        )

        if not compatible:
            # Client reports incompatibility
            error_msg: Dict[str, Any] = {
                "type": "handshake_complete",
                "status": "error",
                "message": "Client reported incompatibility with server requirements",
            }
            await websocket.send(msgpack.packb(error_msg, use_bin_type=True))
            logger.error("Handshake failed: client incompatible")
            return

        # Validate client configuration matches server requirements
        server_caps = self.adapter.get_capabilities()
        client_mode = client_config.get("observation_mode")

        if client_mode and client_mode != server_caps["observation_mode"]:
            error_msg = {
                "type": "handshake_complete",
                "status": "error",
                "message": (
                    f"Observation mode mismatch: server expects "
                    f"'{server_caps['observation_mode']}', client offers '{client_mode}'"
                ),
            }
            await websocket.send(msgpack.packb(error_msg, use_bin_type=True))
            logger.error(f"Handshake failed: observation mode mismatch")
            return

        # Validate num_panos if panoramic mode
        if server_caps["observation_mode"] == "panoramic":
            server_panos = server_caps.get("num_panos")
            client_panos = client_config.get("num_panos")
            if server_panos and client_panos and server_panos != client_panos:
                error_msg = {
                    "type": "handshake_complete",
                    "status": "error",
                    "message": (
                        f"Panoramic view count mismatch: server expects "
                        f"{server_panos}, client offers {client_panos}"
                    ),
                }
                await websocket.send(msgpack.packb(error_msg, use_bin_type=True))
                logger.error(f"Handshake failed: num_panos mismatch")
                return

        # Handshake successful
        success_msg: Dict[str, Any] = {
            "type": "handshake_complete",
            "status": "ok",
            "message": None,
        }
        await websocket.send(msgpack.packb(success_msg, use_bin_type=True))
        self._handshake_complete = True
        logger.info("Handshake complete")

    async def run(self) -> None:
        """Start the WebSocket server.

        Runs indefinitely until interrupted.
        """
        logger.info(f"Starting policy server on {self.host}:{self.port}")
        logger.info(f"Using adapter: {type(self.adapter).__name__}")

        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=100 * 1024 * 1024,  # 100MB max message size
        ):
            logger.info("Server is ready. Waiting for connections...")
            await asyncio.Future()  # Run forever
