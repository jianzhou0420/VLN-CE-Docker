#!/usr/bin/env python3
"""
Seq2Seq Policy Server entry point.

Starts a WebSocket server that hosts a Seq2Seq
navigation policy for VLN-CE evaluation.

Usage:
    python vlnce_server/seq2seq_server.py --port 8765

To test with the eval client:
    # Terminal 1: Start this server
    python vlnce_server/seq2seq_server.py --port 8765

    # Terminal 2: Run eval client
    python docker/vlnce/vlnce_server/ws_eval_client.py \
        --server ws://localhost:8765 --split val_seen
"""

import argparse
import asyncio
import logging

import torch

from vlnce_server.adapters.seq2seq_adapter import Seq2SeqAdapter
from vlnce_server.policy_server import PolicyServer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Seq2Seq Policy Server for VLN-CE Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with default checkpoint
    python vlnce_server/seq2seq_server.py --port 8765

    # Start server with custom checkpoint
    python vlnce_server/seq2seq_server.py --port 8765 \\
        --checkpoint data/checkpoints/my_model.pth

    # Use specific GPU
    python vlnce_server/seq2seq_server.py --port 8765 --gpu 1
        """,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/checkpoints/Seq2Seq_DA.pth",
        help="Path to Seq2Seq model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz",
        help="Path to vocabulary JSON file",
    )
    parser.add_argument(
        "--ddppo-checkpoint",
        type=str,
        default="data/ddppo-models/gibson-2plus-resnet50.pth",
        help="Path to DDPPO depth encoder weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vlnce_baselines/config/r2r_baselines/seq2seq.yaml",
        help="Path to Seq2Seq config YAML file",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--max-instruction-length",
        type=int,
        default=200,
        help="Maximum instruction sequence length (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Seq2Seq server requires GPU.")
        return 1

    device = torch.device(f"cuda:{args.gpu}")
    logger.info(f"Using device: {device}")

    # Create Seq2Seq adapter
    logger.info("Initializing Seq2Seq adapter...")
    adapter = Seq2SeqAdapter(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        ddppo_checkpoint=args.ddppo_checkpoint,
        device=device,
        max_instruction_length=args.max_instruction_length,
        config_path=args.config,
    )

    # Create and run server
    server = PolicyServer(adapter, host=args.host, port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
