"""
Base adapter interface for VLN-CE policy servers.

All model adapters must implement this interface to work with PolicyServer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np


class PolicyAdapter(ABC):
    """Abstract base class for policy adapters.

    Adapters handle model-specific logic including:
    - Loading models from checkpoints
    - Transforming observations to model input format
    - Managing model state (e.g., RNN hidden states)
    - Transforming model output to action format

    The PolicyServer calls these methods without knowing
    anything about the underlying model implementation.
    """

    @abstractmethod
    def on_episode_start(self, episode_id: str, instruction: Dict[str, Any]) -> None:
        """Called when a new episode starts.

        Use this to:
        - Reset model state (RNN hidden states, etc.)
        - Cache/preprocess instruction for the episode

        Args:
            episode_id: Unique identifier for this episode
            instruction: Dict containing instruction data:
                - "text": Natural language instruction string
                - "tokens": Optional pre-tokenized token IDs
                - "trajectory_id": Trajectory identifier
        """
        pass

    @abstractmethod
    def on_observation(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        instruction: Dict[str, Any],
    ) -> Union[int, Dict[str, Any]]:
        """Process observation and return action.

        Steps typically include:
        1. Transform inputs (numpy arrays -> model tensors)
        2. Run model inference
        3. Update internal state (prev_action, RNN states)
        4. Transform output (model tensor -> action)

        Args:
            rgb: RGB image array, shape (H, W, 3) for egocentric or
                 (num_panos, H, W, 3) for panoramic, dtype uint8
            depth: Depth image array, shape (H, W, 1) for egocentric or
                   (num_panos, H, W, 1) for panoramic, dtype float32
            instruction: Dict containing instruction data (same as on_episode_start)

        Returns:
            For discrete action models (CMA, Seq2Seq):
                Action integer:
                    0 = STOP
                    1 = MOVE_FORWARD
                    2 = TURN_LEFT
                    3 = TURN_RIGHT

            For waypoint models (HPN, WPN):
                Action dict:
                    {"action": "STOP"} or
                    {"action": "GO_TOWARD_POINT", "action_args": {"r": float, "theta": float}}
        """
        pass

    @abstractmethod
    def on_episode_end(self) -> None:
        """Called when an episode ends.

        Use for cleanup if needed. State will typically be reset
        on the next on_episode_start call anyway.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return adapter capabilities for protocol negotiation.

        Called by PolicyServer during handshake to advertise server
        capabilities to the client. The client uses this to auto-configure
        its observation format and action handling.

        Returns:
            Dict with capability information:
                - "observation_mode": "egocentric" | "panoramic"
                - "action_type": "discrete" | "waypoint"
                - "num_panos": int | None (number of panoramic views, None for egocentric)
                - "rgb_shape": List[int] - expected RGB shape
                - "depth_shape": List[int] - expected depth shape
                - "action_space": Dict with action details
        """
        pass

    @property
    def server_type(self) -> str:
        """Return server type identifier for protocol negotiation.

        Returns:
            Server type string (e.g., "cma", "seq2seq", "waypoint")
        """
        return self.__class__.__name__.replace("Adapter", "").lower()
