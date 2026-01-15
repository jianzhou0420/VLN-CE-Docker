# VLN-CE Policy Server Docker

Standalone Docker container for serving VLN-CE navigation policies via WebSocket.

## Overview

This Docker setup provides:
- WebSocket-based policy servers for CMA, Seq2Seq, and Waypoint models
- Protocol v1.1 with automatic capability negotiation
- Support for both egocentric (CMA, Seq2Seq) and panoramic (Waypoint) observations

## Supported Policies

| Policy | Description | Observation Mode | Action Type |
|--------|-------------|-----------------|-------------|
| CMA | Cross-Modal Attention | Egocentric (single view) | Discrete (0-5) |
| Seq2Seq | Sequence-to-Sequence | Egocentric (single view) | Discrete (0-5) |
| Waypoint | HPN/WPN | Panoramic (12 views) | Waypoint (r, theta) |

## Prerequisites

### Required Data

1. **Model Checkpoints**
   - CMA: `CMA_PM_DA_Aug.pth`
   - Seq2Seq: `Seq2Seq_DA.pth`
   - Waypoint: `HPN.pth` or `WPN.pth`

2. **Vocabulary Files**
   ```
   data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
   ```

3. **DDPPO Depth Encoder**
   ```bash
   wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip
   unzip ddppo-models.zip -d data/
   ```

### System Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 11.1+ support
- At least 8GB GPU memory

## Quick Start

### 1. Download Data

Use the setup script to download checkpoints, vocabulary, and DDPPO models:

```bash
cd docker/policy_server

# Download everything
./setup_data.sh

# Or download specific components
./setup_data.sh cma       # CMA checkpoint only
./setup_data.sh seq2seq   # Seq2Seq checkpoint only
./setup_data.sh waypoint  # HPN/WPN checkpoints
./setup_data.sh vocab     # Vocabulary files
./setup_data.sh ddppo     # DDPPO depth encoder
```

This creates the following structure:
```
data/
├── checkpoints/
│   ├── CMA_PM_DA_Aug.pth      # CMA checkpoint (141MB)
│   ├── Seq2Seq_DA.pth         # Seq2Seq checkpoint (135MB)
│   ├── HPN.pth                # HPN waypoint checkpoint (97MB)
│   └── WPN.pth                # WPN waypoint checkpoint (97MB)
├── datasets/
│   └── R2R_VLNCE_v1-3_preprocessed/
│       └── train/
│           └── train.json.gz  # Vocabulary file
└── ddppo-models/
    └── gibson-2plus-resnet50.pth  # Depth encoder (672MB)
```

### 2. Build the Docker Image

```bash
cd docker/policy_server
docker build -t vlnce-policy-server .
```

### 3. Run a Server

**Using Docker directly:**
```bash
# CMA server
docker run --rm -it --gpus all --network=host \
  -v $(pwd)/data/checkpoints:/app/data/checkpoints:ro \
  -v $(pwd)/data/datasets:/app/data/datasets:ro \
  -v $(pwd)/data/ddppo-models:/app/data/ddppo-models:ro \
  vlnce-policy-server cma --port 8765

# Seq2Seq server
docker run --rm -it --gpus all --network=host \
  -v $(pwd)/data/checkpoints:/app/data/checkpoints:ro \
  -v $(pwd)/data/datasets:/app/data/datasets:ro \
  -v $(pwd)/data/ddppo-models:/app/data/ddppo-models:ro \
  vlnce-policy-server seq2seq --port 8765

# Waypoint server
docker run --rm -it --gpus all --network=host \
  -v $(pwd)/data/checkpoints:/app/data/checkpoints:ro \
  -v $(pwd)/data/datasets:/app/data/datasets:ro \
  -v $(pwd)/data/ddppo-models:/app/data/ddppo-models:ro \
  vlnce-policy-server waypoint --port 8765
```

**Using Docker Compose:**
```bash
# CMA server
docker compose --profile cma up

# Seq2Seq server
docker compose --profile seq2seq up

# Waypoint server
docker compose --profile waypoint up
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POLICY_PORT` | 8765 | WebSocket server port |
| `POLICY_GPU` | 0 | GPU device ID |
| `CHECKPOINT_DIR` | ./data/checkpoints | Checkpoint directory mount |
| `VOCAB_DIR` | ./data/datasets | Vocabulary directory mount |
| `DDPPO_DIR` | ./data/ddppo-models | DDPPO models mount |
| `CMA_CHECKPOINT` | /app/data/checkpoints/CMA_PM_DA_Aug.pth | CMA model path |
| `SEQ2SEQ_CHECKPOINT` | /app/data/checkpoints/Seq2Seq_DA.pth | Seq2Seq model path |
| `WAYPOINT_CHECKPOINT` | /app/data/checkpoints/HPN.pth | Waypoint model path |
| `VOCAB_PATH` | .../train/train.json.gz | Vocabulary file path |
| `DDPPO_CHECKPOINT` | .../gibson-2plus-resnet50.pth | DDPPO weights path |
| `NUM_PANOS` | 12 | Panoramic views (waypoint only) |

## Command-Line Options

```
Usage: docker run vlnce-policy-server <server_type> [options]

Server types:
  cma       - CMA (Cross-Modal Attention) policy server
  seq2seq   - Seq2Seq policy server
  waypoint  - Waypoint (HPN/WPN) policy server

Common options:
  --host HOST          Server host (default: 0.0.0.0)
  --port PORT          Server port (default: 8765)
  --checkpoint PATH    Model checkpoint path
  --vocab PATH         Vocabulary file path
  --ddppo-checkpoint   DDPPO depth encoder weights
  --config PATH        Config YAML file
  --gpu GPU_ID         GPU device ID (default: 0)
  -v, --verbose        Enable verbose logging

Waypoint-specific options:
  --num-panos N        Number of panoramic views (default: 12)
```

## Protocol

The server implements WebSocket protocol v1.1 with:
- **Automatic capability negotiation** via server_hello/client_hello handshake
- **msgpack serialization** with numpy support
- **Observation modes**: egocentric (CMA, Seq2Seq) or panoramic (Waypoint)

See [../vlnce/protocol.md](../vlnce/protocol.md) for full protocol specification.

### Action Space

**Discrete Actions (CMA, Seq2Seq):**
| Index | Action | Description |
|-------|--------|-------------|
| 0 | STOP | End the episode |
| 1 | MOVE_FORWARD | Move forward 0.25m |
| 2 | TURN_LEFT | Turn left 15 degrees |
| 3 | TURN_RIGHT | Turn right 15 degrees |
| 4 | LOOK_UP | Look up 15 degrees |
| 5 | LOOK_DOWN | Look down 15 degrees |

**Waypoint Actions (HPN/WPN):**
```json
{"action": "STOP"}
{"action": "GO_TOWARD_POINT", "action_args": {"r": 2.5, "theta": 0.52}}
```

## Testing with Evaluation Client

Start the policy server and connect with the evaluation client:

```bash
# Terminal 1: Start policy server
cd docker/policy_server
docker compose --profile cma up

# Terminal 2: Run evaluation client
cd docker/vlnce
VLNCE_SERVER=ws://localhost:8765 docker compose up
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:11.1.1-base nvidia-smi
```

### Checkpoint Not Found
- Verify checkpoint files are in the mounted directory
- Check file permissions allow read access
- Confirm volume mount paths are correct

### Connection Refused
- Ensure policy server is running before starting client
- Check firewall allows connections on the server port
- Verify `--network=host` is set

### Out of Memory
- CMA/Seq2Seq: ~4GB GPU memory
- Waypoint: ~6GB GPU memory
- Reduce batch size or use smaller model if needed
