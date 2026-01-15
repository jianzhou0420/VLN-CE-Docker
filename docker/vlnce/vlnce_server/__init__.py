"""
VLN-CE WebSocket Evaluation Server

This module provides WebSocket-based evaluation for VLN-CE,
allowing external policy servers to control navigation agents.

Components:
    - ws_eval_client: Evaluation client that connects to external policy servers
    - example_policy_server: Example policy server implementation for testing
"""

from vlnce_server.ws_eval_client import (
    ClientConfig,
    VLNCEEvaluationClient,
)

__all__ = [
    "ClientConfig",
    "VLNCEEvaluationClient",
]
