"""
Seq2Seq adapter for VLN-CE policy server.

Handles loading the Seq2Seq model, managing RNN state, and transforming
observations for inference.
"""

import logging
from typing import Any, Dict

import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_server.base_adapter import PolicyAdapter
from vlnce_server.transforms.tokenizer import InstructionTokenizer

logger = logging.getLogger(__name__)


class Seq2SeqAdapter(PolicyAdapter):
    """Adapter for Seq2Seq policy.

    Handles:
    - Loading Seq2Seq model from checkpoint
    - Tokenizing instructions
    - Managing RNN hidden states across steps
    - Transforming observations to model input format

    Key differences from CMA:
    - Simpler architecture without cross-modal attention
    - Single RNN layer (vs 2 for CMA)
    - Instruction encoder: bidirectional=False, final_state_only=True
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        ddppo_checkpoint: str,
        device: torch.device,
        max_instruction_length: int = 200,
        config_path: str = "vlnce_baselines/config/r2r_baselines/seq2seq.yaml",
    ):
        """Initialize Seq2Seq adapter.

        Args:
            checkpoint_path: Path to Seq2Seq model checkpoint (.pth file)
            vocab_path: Path to vocabulary JSON file (train.json.gz)
            ddppo_checkpoint: Path to DDPPO depth encoder weights
            device: Torch device for inference
            max_instruction_length: Maximum instruction sequence length
            config_path: Path to Seq2Seq config YAML file
        """
        self.device = device
        self.max_instruction_length = max_instruction_length
        self.checkpoint_path = checkpoint_path
        self.ddppo_checkpoint = ddppo_checkpoint
        self.config_path = config_path

        # Initialize tokenizer
        logger.info(f"Loading vocabulary from {vocab_path}")
        self.tokenizer = InstructionTokenizer(vocab_path)
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")

        # Load Seq2Seq model
        logger.info(f"Loading Seq2Seq model from {checkpoint_path}")
        self.policy = self._load_model()
        self.policy.eval()
        logger.info("Seq2Seq model loaded successfully")

        # Initialize state variables
        self._init_state_variables()

        # Current instruction tokens (cached per episode)
        self.current_instruction_tokens: np.ndarray = None
        self.current_episode_id: str = None

    def _load_model(self):
        """Load Seq2Seq model from checkpoint.

        Returns:
            Loaded Seq2Seq policy
        """
        # Create config
        config = self._create_config()

        # Create observation and action spaces
        observation_space = self._create_observation_space()
        action_space = spaces.Discrete(4)  # STOP, FORWARD, LEFT, RIGHT

        # Initialize policy
        policy_cls = baseline_registry.get_policy(config.MODEL.policy_name)
        policy = policy_cls.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Load checkpoint weights
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        policy.load_state_dict(ckpt["state_dict"])
        policy.to(self.device)

        return policy

    def _create_config(self):
        """Create Seq2Seq configuration.

        Returns:
            Config object with Seq2Seq settings
        """
        from vlnce_baselines.config.default import get_config

        # Load base config from Seq2Seq yaml
        config = get_config(self.config_path)

        # Override settings for inference
        config.defrost()

        # Seq2Seq-specific settings (NOT bidirectional, final state only)
        config.MODEL.INSTRUCTION_ENCODER.bidirectional = False
        config.MODEL.INSTRUCTION_ENCODER.final_state_only = True

        # Set depth encoder checkpoint
        config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = self.ddppo_checkpoint

        # Disable progress monitor for inference
        config.MODEL.PROGRESS_MONITOR.use = False

        config.freeze()
        return config

    def _create_observation_space(self) -> spaces.Dict:
        """Create observation space for Seq2Seq model.

        Returns:
            Gym Dict observation space
        """
        return spaces.Dict({
            "rgb": spaces.Box(
                low=0, high=255,
                shape=(224, 224, 3),
                dtype=np.uint8
            ),
            "depth": spaces.Box(
                low=0.0, high=1.0,
                shape=(256, 256, 1),
                dtype=np.float32
            ),
            "instruction": spaces.Box(
                low=0, high=self.tokenizer.vocab_size,
                shape=(self.max_instruction_length,),
                dtype=np.int64
            ),
        })

    def _init_state_variables(self) -> None:
        """Initialize RNN state variables."""
        self.hidden_size = 512  # STATE_ENCODER.hidden_size
        self.num_layers = self.policy.net.num_recurrent_layers
        logger.info(f"RNN state: {self.num_layers} layers, {self.hidden_size} hidden size")
        self._reset_rnn_state()

    def _reset_rnn_state(self) -> None:
        """Reset RNN hidden state for new episode."""
        self.rnn_states = torch.zeros(
            1,  # batch size
            self.num_layers,
            self.hidden_size,
            device=self.device,
        )
        self.prev_action = torch.zeros(
            1, 1,
            device=self.device,
            dtype=torch.long,
        )
        # mask=0 resets RNN state, mask=1 keeps it
        # Must be bool for torch.where in RNN state encoder
        self.mask = torch.zeros(1, 1, device=self.device, dtype=torch.bool)

    # === PolicyAdapter Interface ===

    def on_episode_start(self, episode_id: str, instruction: Dict[str, Any]) -> None:
        """Reset state and tokenize instruction for new episode.

        Args:
            episode_id: Episode identifier
            instruction: Dict with 'text' key containing instruction string
        """
        self.current_episode_id = episode_id
        self._reset_rnn_state()

        # Tokenize and cache instruction
        text = instruction.get("text", "")
        self.current_instruction_tokens = self.tokenizer.tokenize(
            text, self.max_instruction_length
        )

        logger.debug(f"Episode {episode_id}: tokenized instruction ({len(text)} chars)")

    def on_observation(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        instruction: Dict[str, Any],
    ) -> int:
        """Transform observation, run model inference, return action.

        Args:
            rgb: RGB image (224, 224, 3) uint8
            depth: Depth image (256, 256, 1) float32
            instruction: Instruction dict (already cached, not used here)

        Returns:
            Action integer (0-3)
        """
        # Input transform: numpy -> tensor batch
        batch = self._transform_input(rgb, depth)

        # Model inference
        with torch.no_grad():
            action, self.rnn_states = self.policy.act(
                batch,
                self.rnn_states,
                self.prev_action,
                self.mask,
                deterministic=True,
            )

        # Update state for next step
        self.prev_action = action
        self.mask = torch.ones(1, 1, device=self.device, dtype=torch.bool)  # Keep RNN state

        # Output transform: tensor -> int
        return action.item()

    def on_episode_end(self) -> None:
        """Called when episode ends.

        State will be reset on next on_episode_start.
        """
        logger.debug(f"Episode {self.current_episode_id} ended")

    # === Input Transform ===

    def _transform_input(
        self, rgb: np.ndarray, depth: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Transform numpy observations to model input tensors.

        Args:
            rgb: RGB image array
            depth: Depth image array

        Returns:
            Dict of batched tensors on device
        """
        return {
            "rgb": torch.from_numpy(rgb).unsqueeze(0).to(self.device),
            "depth": torch.from_numpy(depth).unsqueeze(0).to(self.device),
            "instruction": torch.from_numpy(
                self.current_instruction_tokens
            ).unsqueeze(0).to(self.device),
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return adapter capabilities for protocol negotiation.

        Returns:
            Dict with egocentric/discrete configuration
        """
        return {
            "observation_mode": "egocentric",
            "action_type": "discrete",
            "num_panos": None,
            "rgb_shape": [224, 224, 3],
            "depth_shape": [256, 256, 1],
            "action_space": {
                "type": "discrete",
                "num_actions": 4,
                "actions": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
            },
        }
