#!/usr/bin/env python3
"""
Waypoint (HPN/WPN) Policy Server for VLN-CE.

Starts a WebSocket server that serves waypoint model predictions.
Requires panoramic observations (12 views) from the eval client.

Usage:
    python vlnce_server/waypoint_server.py \
        --checkpoint data/checkpoints/HPN.pth \
        --vocab data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
        --ddppo-checkpoint data/ddppo-models/gibson-2plus-resnet50.pth \
        --port 8765
"""

import argparse
import asyncio
import logging

import torch

from vlnce_server.adapters.waypoint_adapter import WaypointAdapter
from vlnce_server.policy_server import PolicyServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Waypoint (HPN/WPN) Policy Server for VLN-CE"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/HPN.pth",
        help="Path to waypoint model checkpoint"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary file (train.json.gz)"
    )
    parser.add_argument(
        "--ddppo-checkpoint",
        type=str,
        default="data/ddppo-models/gibson-2plus-resnet50.pth",
        help="Path to DDPPO depth encoder checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vlnce_baselines/config/r2r_waypoint/5-hpn-_c.yaml",
        help="Path to waypoint config YAML"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--max-instruction-length",
        type=int,
        default=200,
        help="Maximum instruction token length"
    )
    parser.add_argument(
        "--num-panos",
        type=int,
        default=12,
        help="Number of panoramic views"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create waypoint adapter
    logger.info("Initializing WaypointAdapter...")
    adapter = WaypointAdapter(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        ddppo_checkpoint=args.ddppo_checkpoint,
        device=device,
        max_instruction_length=args.max_instruction_length,
        config_path=args.config,
        num_panos=args.num_panos,
    )

    # Create and run server
    server = PolicyServer(adapter, host=args.host, port=args.port)

    logger.info(f"Starting Waypoint server on {args.host}:{args.port}")
    logger.info(f"Model: {args.checkpoint}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Expecting {args.num_panos} panoramic views per observation")

    asyncio.run(server.run())


if __name__ == "__main__":
    main()
