"""
Waypoint (HPN/WPN) adapter for VLN-CE policy server.

Handles loading waypoint models, managing RNN state, and transforming
panoramic observations for inference.
"""

import logging
from numbers import Number
from typing import Any, Dict, Optional

import numpy as np
import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from torch.distributions import constraints
from torch.distributions.normal import Normal

from vlnce_server.base_adapter import PolicyAdapter
from vlnce_server.transforms.tokenizer import InstructionTokenizer

logger = logging.getLogger(__name__)


# Monkey-patch TruncatedNormal to fix PyTorch 1.7+ cdf() requiring Tensor inputs
def _patch_truncated_normal():
    from vlnce_baselines.models import utils as model_utils

    def _patched_init(self, loc, scale, smin=-np.inf, smax=np.inf, validate_args=None):
        torch.nn.Module.__init__(self)
        assert smin < smax, "smin must be less than smax"
        assert np.isfinite(smin) and np.isfinite(smax), \
            "two-sided truncation is required. Set both `smin` and `smax`."
        assert (loc >= smin).all() and (loc <= smax).all(), \
            f"loc is out of range ({smin}, {smax}): {loc}"
        if isinstance(scale, Number):
            assert scale >= 0.0, "scale is negative"
        else:
            assert (scale >= 0.0).all(), "scale is negative"

        self._normal = Normal(loc, scale, validate_args=validate_args)
        self._loc = loc
        self._scale = scale
        self._smin = smin
        self._smax = smax
        self._unbounded = smin == -np.inf and smax == np.inf
        self.A = 1 / (self._scale * np.sqrt(2 * np.pi))
        # Fix: convert floats to tensors for PyTorch 1.7+ cdf() compatibility
        smax_t = torch.tensor(smax, dtype=loc.dtype, device=loc.device)
        smin_t = torch.tensor(smin, dtype=loc.dtype, device=loc.device)
        self.Z = self._normal.cdf(smax_t) - self._normal.cdf(smin_t)
        self.support = constraints.interval(smin, smax)
        self._init_mean_variance_entropy()

    model_utils.TruncatedNormal.__init__ = _patched_init
    logger.info("Patched TruncatedNormal for PyTorch 1.7+ compatibility")


_patch_truncated_normal()

# Constants matching waypoint_predictors.py
ANGLE_FEATURE_SIZE = 4
DEFAULT_NUM_PANOS = 12


class WaypointAdapter(PolicyAdapter):
    """Adapter for Waypoint (HPN/WPN) policy.

    Handles:
    - Loading waypoint model from checkpoint
    - Tokenizing instructions
    - Managing RNN hidden states across steps
    - Managing visual history (rgb_history, depth_history)
    - Transforming panoramic observations to model input format
    - Converting waypoint outputs to action format
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        ddppo_checkpoint: str,
        device: torch.device,
        max_instruction_length: int = 200,
        config_path: str = "vlnce_baselines/config/r2r_waypoint/5-hpn-_c.yaml",
        num_panos: int = DEFAULT_NUM_PANOS,
    ):
        """Initialize Waypoint adapter.

        Args:
            checkpoint_path: Path to waypoint model checkpoint (.pth file)
            vocab_path: Path to vocabulary JSON file (train.json.gz)
            ddppo_checkpoint: Path to DDPPO depth encoder weights
            device: Torch device for inference
            max_instruction_length: Maximum instruction sequence length
            config_path: Path to waypoint config YAML file
            num_panos: Number of panoramic views (default 12)
        """
        self.device = device
        self.max_instruction_length = max_instruction_length
        self.checkpoint_path = checkpoint_path
        self.ddppo_checkpoint = ddppo_checkpoint
        self.config_path = config_path
        self.num_panos = num_panos

        # Initialize tokenizer
        logger.info(f"Loading vocabulary from {vocab_path}")
        self.tokenizer = InstructionTokenizer(vocab_path)
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")

        # Precompute angle features for panoramic views
        self.angle_features = self._create_angle_features()

        # Load waypoint model
        logger.info(f"Loading Waypoint model from {checkpoint_path}")
        self.policy = self._load_model()
        self.policy.eval()
        logger.info("Waypoint model loaded successfully")

        # Initialize state variables
        self._init_state_variables()

        # Current instruction tokens (cached per episode)
        self.current_instruction_tokens: Optional[np.ndarray] = None
        self.current_episode_id: Optional[str] = None

    def _create_angle_features(self) -> np.ndarray:
        """Create angle features for panoramic views.

        Returns:
            Array of shape [num_panos, 4] with [sin(θ), cos(θ), 0, 1] for each view
        """
        angle_features = []
        for i in range(self.num_panos):
            angle = np.pi * 2 / self.num_panos * i
            angle_features.append([np.sin(angle), np.cos(angle), 0.0, 1.0])
        return np.array(angle_features, dtype=np.float32)

    def _load_model(self):
        """Load waypoint model from checkpoint.

        Returns:
            Loaded waypoint policy
        """
        # Create config
        config = self._create_config()

        # Create observation and action spaces
        observation_space = self._create_observation_space()
        # Waypoint uses continuous action space but the dimension is ignored
        action_space = spaces.Discrete(1)

        # Initialize policy
        policy_cls = baseline_registry.get_policy(config.MODEL.policy_name)
        policy = policy_cls.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Load checkpoint weights
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")

        # Handle DDPPO checkpoint format which wraps policy in actor_critic
        state_dict = ckpt["state_dict"]
        if any(k.startswith("actor_critic.") for k in state_dict.keys()):
            # Strip actor_critic. prefix from keys
            state_dict = {
                k.replace("actor_critic.", ""): v
                for k, v in state_dict.items()
                if k.startswith("actor_critic.")
            }
            logger.info("Converted DDPPO checkpoint format (stripped actor_critic. prefix)")

        policy.load_state_dict(state_dict)
        policy.to(self.device)

        return policy

    def _create_config(self):
        """Create waypoint configuration.

        Returns:
            Config object with waypoint settings
        """
        from vlnce_baselines.config.default import get_config

        # Load base config from waypoint yaml
        config = get_config(self.config_path)

        # Override settings for inference
        config.defrost()

        # Set depth encoder checkpoint
        config.MODEL.DEPTH_ENCODER.ddppo_checkpoint = self.ddppo_checkpoint

        # Set number of panoramic views
        config.TASK_CONFIG.TASK.PANO_ROTATIONS = self.num_panos

        config.freeze()
        return config

    def _create_observation_space(self) -> spaces.Dict:
        """Create observation space for waypoint model.

        Returns:
            Gym Dict observation space
        """
        return spaces.Dict({
            # Panoramic RGB: [num_panos, H, W, 3]
            "rgb": spaces.Box(
                low=0, high=255,
                shape=(self.num_panos, 224, 224, 3),
                dtype=np.uint8
            ),
            # Panoramic Depth: [num_panos, H, W, 1]
            "depth": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_panos, 256, 256, 1),
                dtype=np.float32
            ),
            # Instruction tokens
            "instruction": spaces.Box(
                low=0, high=self.tokenizer.vocab_size,
                shape=(self.max_instruction_length,),
                dtype=np.int64
            ),
            # RGB history (single frame)
            "rgb_history": spaces.Box(
                low=0, high=255,
                shape=(224, 224, 3),
                dtype=np.uint8
            ),
            # Depth history (single frame)
            "depth_history": spaces.Box(
                low=0.0, high=1.0,
                shape=(256, 256, 1),
                dtype=np.float32
            ),
            # Angle features for panoramic views
            "angle_features": spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_panos, ANGLE_FEATURE_SIZE),
                dtype=np.float32
            ),
        })

    def _init_state_variables(self) -> None:
        """Initialize RNN state variables and history buffers."""
        self.hidden_size = 256  # STATE_ENCODER.hidden_size from config
        self.num_layers = self.policy.net.num_recurrent_layers
        logger.info(f"RNN state: {self.num_layers} layers, {self.hidden_size} hidden size")

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset RNN hidden state and history buffers for new episode."""
        # RNN states
        self.rnn_states = torch.zeros(
            1,  # batch size
            self.num_layers,
            self.hidden_size,
            device=self.device,
        )

        # Previous actions (dict format for waypoint)
        self.prev_actions = {
            "pano": torch.zeros(1, 1, device=self.device, dtype=torch.long),
            "offset": torch.zeros(1, 1, device=self.device, dtype=torch.float32),
            "distance": torch.zeros(1, 1, device=self.device, dtype=torch.float32),
        }

        # Mask: 0 resets RNN state, 1 keeps it
        self.mask = torch.zeros(1, 1, device=self.device, dtype=torch.bool)

        # Visual history buffers (initialized to zeros)
        self.rgb_history = np.zeros((224, 224, 3), dtype=np.uint8)
        self.depth_history = np.zeros((256, 256, 1), dtype=np.float32)

    # === PolicyAdapter Interface ===

    def on_episode_start(self, episode_id: str, instruction: Dict[str, Any]) -> None:
        """Reset state and tokenize instruction for new episode.

        Args:
            episode_id: Episode identifier
            instruction: Dict with 'text' key containing instruction string
        """
        self.current_episode_id = episode_id
        self._reset_state()

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
    ) -> Dict[str, Any]:
        """Transform observation, run model inference, return action.

        Args:
            rgb: Panoramic RGB images [num_panos, 224, 224, 3] uint8
            depth: Panoramic depth images [num_panos, 256, 256, 1] float32
            instruction: Instruction dict (already cached, not used here)

        Returns:
            Action dict: {"action": "STOP"} or
                        {"action": "GO_TOWARD_POINT", "action_args": {"r": float, "theta": float}}
        """
        # Input transform: numpy -> tensor batch
        batch = self._transform_input(rgb, depth)

        # Model inference
        with torch.no_grad():
            (
                value,
                actions,
                action_elements,
                modes,
                variances,
                action_log_probs,
                rnn_states_out,
                pano_stop_distribution,
            ) = self.policy.act(
                batch,
                self.rnn_states,
                self.prev_actions,
                self.mask,
                deterministic=True,
            )

        # Update state for next step
        self.rnn_states = rnn_states_out
        self.prev_actions = {
            "pano": action_elements["pano"],
            "offset": action_elements["offset"],
            "distance": action_elements["distance"],
        }
        self.mask = torch.ones(1, 1, device=self.device, dtype=torch.bool)

        # Update visual history with selected pano view
        selected_pano = action_elements["pano"].item() % self.num_panos
        self.rgb_history = rgb[selected_pano].copy()
        self.depth_history = depth[selected_pano].copy()

        # Output: actions[0] is already in the correct format
        action = actions[0]

        # Normalize action format for server
        if action.get("action") == "STOP":
            return {"action": "STOP"}
        else:
            # action["action"] is a dict with action details
            inner_action = action["action"]
            return {
                "action": "GO_TOWARD_POINT",
                "action_args": inner_action["action_args"]
            }

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
            rgb: Panoramic RGB array [num_panos, 224, 224, 3]
            depth: Panoramic depth array [num_panos, 256, 256, 1]

        Returns:
            Dict of batched tensors on device
        """
        return {
            # Add batch dimension: [1, num_panos, H, W, C]
            "rgb": torch.from_numpy(rgb).unsqueeze(0).to(self.device),
            "depth": torch.from_numpy(depth).unsqueeze(0).to(self.device),
            "instruction": torch.from_numpy(
                self.current_instruction_tokens
            ).unsqueeze(0).to(self.device),
            # History needs batch dimension: [H, W, C] -> [1, H, W, C]
            "rgb_history": torch.from_numpy(self.rgb_history.copy()).unsqueeze(0).to(self.device),
            "depth_history": torch.from_numpy(self.depth_history.copy()).unsqueeze(0).to(self.device),
            # Angle features need batch dimension: [num_panos, 4] -> [1, num_panos, 4]
            "angle_features": torch.from_numpy(self.angle_features).unsqueeze(0).to(self.device),
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Return adapter capabilities for protocol negotiation.

        Returns:
            Dict with panoramic/waypoint configuration
        """
        return {
            "observation_mode": "panoramic",
            "action_type": "waypoint",
            "num_panos": self.num_panos,
            "rgb_shape": [self.num_panos, 224, 224, 3],
            "depth_shape": [self.num_panos, 256, 256, 1],
            "action_space": {
                "type": "continuous",
                "actions": ["STOP", "GO_TOWARD_POINT"],
            },
        }
