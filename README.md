# VLN-CE-Docker

A Dockerized [VLN-CE](https://github.com/jacobkrantz/VLN-CE) evaluation client. Inspired by [OpenPI](https://github.com/Physical-Intelligence/openpi)'s Docker eval practice. Docker isolates the benchmark/simulator environment, avoiding dependency conflicts with your model.

This project decouples the tightly-coupled VLN-CE code. The evaluation client now only produces raw observations and receives native [habitat-sim](https://github.com/facebookresearch/habitat-sim) actions.

The Policy Server uses a two-layer abstraction: `PolicyServer` handles WebSocket communication, `Adapter` handles model input/output conversion. See `base_adapter.py` for the abstract base class.

**Note:** This project follows the minimal changes principle. Results on VLN-CE's three baselines are consistent with the original, but the code was largely written with [Claude Code](https://github.com/anthropics/claude-code) and has not been thoroughly reviewed. If you find bugs, please report them in the issues.


## Architecture

```
┌─────────────────────────┐         WebSocket          ┌─────────────────────────┐
│   Evaluation Client     │◄──────────────────────────►│    Policy Server        │
│   (docker/vlnce)        │                            │  (docker/policy_server) │
├─────────────────────────┤                            ├─────────────────────────┤
│ • Habitat Simulator     │    Observations (RGB,      │ • CMA Policy            │
│ • Matterport3D Scenes   │    Depth, Instruction)     │ • Seq2Seq Policy        │
│ • R2R-CE Dataset        │ ────────────────────────►  │ • Waypoint Policy       │
│ • Metrics Collection    │                            │                         │
│                         │    Actions (0-5 or         │ • GPU Inference         │
│                         │    waypoint coords)        │ • Model Checkpoints     │
│                         │ ◄────────────────────────  │                         │
└─────────────────────────┘                            └─────────────────────────┘
```

Both containers run on the same machine via localhost.


## Version Info

Original VLN-CE used CUDA 10.2 + PyTorch 1.6.0, which doesn't work on modern GPUs. This project upgrades to CUDA 11.1 + PyTorch 1.9.0 with minimal changes.

| Component | Original | This Project |
|-----------|----------|--------------|
| CUDA | 10.2 | 11.1.1 |
| PyTorch | 1.6.0 | 1.9.0 |
| habitat-sim | 0.1.7 | 0.1.7 |
| Python | 3.8 | 3.8.15 |


## Quick Start

```bash
# Clone and init submodules
git clone https://github.com/jianzhou0420/VLN-CE-Docker.git
cd VLN-CE-Docker
git submodule update --init --recursive

# Download data (interactive menu)
./setup_data.sh

# Build containers
cd docker/vlnce && docker build -t vlnce-client .
cd docker/policy_server && docker build -t vlnce-policy-server .

# Run (Terminal 1: policy server)
cd docker/policy_server && docker compose --profile cma up

# Run (Terminal 2: eval client)
cd docker/vlnce && VLNCE_SERVER=ws://localhost:8765 docker compose up
```


## Data Setup

```bash
./setup_data.sh                    # Interactive menu
./setup_data.sh --all              # Download everything
./setup_data.sh --policy cma       # CMA checkpoint only
./setup_data.sh --status           # Check what's downloaded
```

[Matterport3D](https://niessner.github.io/Matterport/) scenes must be downloaded manually (requires access request).


## Supported Policies

| Policy | Observation | Action |
|--------|-------------|--------|
| CMA | Egocentric | Discrete (0-5) |
| Seq2Seq | Egocentric | Discrete (0-5) |
| Waypoint (HPN/WPN) | Panoramic (12 views) | Waypoint (r, theta) |


## Action Space

| ID | Action | Effect |
|----|--------|--------|
| 0 | STOP | End episode |
| 1 | MOVE_FORWARD | 0.25m forward |
| 2 | TURN_LEFT | 15° left |
| 3 | TURN_RIGHT | 15° right |
| 4 | LOOK_UP | 15° up |
| 5 | LOOK_DOWN | 15° down |


## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLNCE_SERVER` | `ws://localhost:8765` | Policy server URI |
| `VLNCE_SPLIT` | `val_seen` | Dataset split |
| `VLNCE_EPISODE_LIMIT` | `-1` | Max episodes (-1 = all) |
| `POLICY_PORT` | `8765` | Server port |
| `POLICY_GPU` | `0` | GPU device ID |


## Protocol

WebSocket with msgpack serialization. Message flow:
1. `server_hello` / `client_hello` - Capability negotiation
2. `episode_start` - New episode with instruction
3. `observation` / `action` - Step loop
4. `episode_end` - Episode metrics
5. `evaluation_complete` - Final aggregated metrics

Full spec: [docker/vlnce/protocol.md](docker/vlnce/protocol.md)


## Results

Saved to `docker/vlnce/data/eval_results/`:

```json
{
  "split": "val_seen",
  "total_episodes": 100,
  "aggregated_metrics": {"success": 0.32, "spl": 0.28, "ndtw": 0.45}
}
```


## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 11.1+ support
- 16GB+ RAM, 8GB+ GPU memory


## Troubleshooting

**GPU not detected:**
```bash
docker run --rm --gpus all nvidia/cuda:11.1.1-base nvidia-smi
```

**Submodule issues:**
```bash
git submodule update --init --recursive --force
```

**Connection refused:** Start policy server before eval client. Check firewall allows port 8765.


## References

- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- [habitat-lab](https://github.com/facebookresearch/habitat-lab)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
