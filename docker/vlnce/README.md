# VLN-CE Docker Environment

Docker container for running the VLN-CE (Vision-and-Language Navigation in Continuous Environments) evaluation client.

## Overview

This Docker setup provides:
- A containerized Habitat simulator environment with Matterport3D scene support
- WebSocket-based evaluation client that connects to external policy servers
- Example policy server for testing (random policy)

## Prerequisites

### Required Data

1. **Matterport3D Scenes** (requires access request)
   - Request access at: https://niessner.github.io/Matterport/
   - Download scene files to `data/scene_datasets/mp3d/`
   - Structure: `data/scene_datasets/mp3d/{scene_id}/{scene_id}.glb`

2. **R2R-CE Dataset**
   ```bash
   # Download R2R_VLNCE dataset
   gdown https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr
   unzip R2R_VLNCE_v1-3_preprocessed.zip -d data/datasets/
   ```

3. **Depth Encoder Models** (optional, for trained policies)
   ```bash
   wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip
   unzip ddppo-models.zip -d data/
   ```

### System Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 11.1+ support
- At least 16GB RAM (32GB recommended for multiple scenes)

## Quick Start

### 1. Initialize Git Submodules

```bash
cd /path/to/VLN-CE
git submodule update --init --recursive
```

### 2. Build the Docker Image

```bash
docker build -t vlnce -f docker/vlnce/Dockerfile .
```

### 3. Run with Docker Compose

```bash
# Run evaluation with the example random policy server
docker compose -f docker/vlnce/compose.yml up --build

# Run with custom settings
VLNCE_SPLIT=val_unseen VLNCE_EPISODE_LIMIT=10 \
  docker compose -f docker/vlnce/compose.yml up
```

## Usage

### Docker Compose (Recommended)

```bash
# Full evaluation with example policy server
docker compose -f docker/vlnce/compose.yml up

# Only the evaluation client (connect to external server)
VLNCE_SERVER=ws://your-server:8765 \
  docker compose -f docker/vlnce/compose.yml --profile standalone up runtime-standalone

# Custom evaluation settings
VLNCE_SPLIT=val_unseen \
VLNCE_EPISODE_LIMIT=100 \
  docker compose -f docker/vlnce/compose.yml up
```

### Docker Run (Manual)

```bash
# Start policy server
docker run --rm -d --name vlnce-server --network=host vlnce \
  --entrypoint python vlnce_server/example_policy_server.py --port 8765

# Run evaluation client
docker run --rm -it --gpus all --network=host \
  -v $(pwd)/data/scene_datasets:/app/data/scene_datasets:ro \
  -v $(pwd)/data/datasets:/app/data/datasets:ro \
  -v $(pwd)/data/eval_results:/app/data/eval_results \
  vlnce --server ws://localhost:8765 --split val_seen --episode-limit 10
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLNCE_SERVER` | `ws://localhost:8765` | Policy server WebSocket URI |
| `VLNCE_SPLIT` | `val_seen` | Dataset split (`val_seen`, `val_unseen`, `test`) |
| `VLNCE_EPISODE_LIMIT` | `-1` | Max episodes (-1 for all) |
| `VLNCE_POLICY` | `random` | Policy type for example server (`random`, `forward`) |

## Client Arguments

```
usage: ws_eval_client.py [-h] --server SERVER [--split {val_seen,val_unseen,test}]
                         [--config CONFIG] [--results-dir RESULTS_DIR]
                         [--episode-limit EPISODE_LIMIT] [--gpu-id GPU_ID]
                         [--timeout TIMEOUT] [--verbose]

Arguments:
  --server SERVER       WebSocket URI of the policy server (required)
  --split SPLIT         Dataset split to evaluate (default: val_seen)
  --config CONFIG       Path to habitat config YAML
  --results-dir DIR     Directory to save results (default: data/eval_results)
  --episode-limit N     Limit number of episodes (-1 for all)
  --gpu-id ID           GPU device ID (default: 0)
  --timeout SECONDS     Action timeout (default: 300)
  --verbose, -v         Enable verbose logging
```

## WebSocket Protocol

See [protocol.md](protocol.md) for detailed protocol documentation.

### Quick Reference

**Message Types:**
- `episode_start`: Sent at episode beginning with instruction
- `observation`: RGB (256x256x3), depth (256x256x1), instruction
- `action`: Server response with action index (0-5)
- `evaluation_complete`: Final metrics after all episodes

**Action Space:**
| Index | Action |
|-------|--------|
| 0 | STOP |
| 1 | MOVE_FORWARD (0.25m) |
| 2 | TURN_LEFT (15°) |
| 3 | TURN_RIGHT (15°) |
| 4 | LOOK_UP (15°) |
| 5 | LOOK_DOWN (15°) |

## Results

Evaluation results are saved to `data/eval_results/` as JSON:

```json
{
  "split": "val_seen",
  "total_episodes": 100,
  "aggregated_metrics": {
    "success": 0.32,
    "spl": 0.28,
    "ndtw": 0.45,
    "distance_to_goal": 5.2,
    "path_length": 12.3
  },
  "per_episode_metrics": { ... }
}
```

## Implementing a Policy Server

```python
import asyncio
import msgpack
import msgpack_numpy
import websockets

msgpack_numpy.patch()

async def handle_client(websocket):
    async for message in websocket:
        msg = msgpack.unpackb(message, raw=False)

        if msg["type"] == "observation" and not msg["done"]:
            # Extract observations
            rgb = msg["rgb"]          # (256, 256, 3) uint8
            depth = msg["depth"]      # (256, 256, 1) float32
            instruction = msg["instruction"]["text"]

            # Your policy logic here
            action = your_model.predict(rgb, depth, instruction)

            # Send action
            response = {"type": "action", "action": int(action)}
            await websocket.send(msgpack.packb(response, use_bin_type=True))

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 8765, max_size=100*1024*1024):
        await asyncio.Future()

asyncio.run(main())
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:11.1.1-base nvidia-smi
```

### Scene Loading Errors
- Verify Matterport3D scenes are in `data/scene_datasets/mp3d/`
- Check scene has both `.glb` and `.navmesh` files

### Connection Refused
- Ensure policy server is running before starting client
- Check firewall allows connections on port 8765
- Verify `--network=host` is set

### Out of Memory
- Reduce `--episode-limit` to process fewer episodes
- Ensure sufficient GPU memory (8GB+ recommended)
