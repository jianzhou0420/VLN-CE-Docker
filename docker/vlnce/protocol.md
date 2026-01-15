# VLN-CE WebSocket Protocol

This document describes the WebSocket protocol used for communication between the VLN-CE evaluation client and the policy server.

**Protocol Version:** 1.1

## Overview

The VLN-CE evaluation client runs the Habitat simulator with Matterport3D scenes and communicates with an external policy server via WebSocket. The client sends observations (RGB, depth, instruction) and receives navigation actions.

Protocol v1.1 introduces capability negotiation, allowing the server to advertise its requirements and the client to auto-configure accordingly.

## Serialization

Uses **msgpack** with NumPy array support via `msgpack-numpy`.

```python
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

# Serialize
packed = msgpack.packb(data, use_bin_type=True)

# Deserialize
data = msgpack.unpackb(packed, raw=False)
```

## Connection Flow

```
┌────────────────┐                      ┌────────────────┐
│     Client     │                      │  Policy Server │
│  (ws_eval_     │                      │                │
│   client.py)   │                      │                │
└───────┬────────┘                      └───────┬────────┘
        │                                       │
        │──────── WebSocket Connect ───────────►│
        │                                       │
        │◄─────── server_hello ────────────────│  ← Protocol v1.1
        │       (capabilities)                  │
        │                                       │
        │─────── client_hello ─────────────────►│  ← Protocol v1.1
        │       (configuration)                 │
        │                                       │
        │◄───── handshake_complete ────────────│  ← Protocol v1.1
        │                                       │
   ┌────┴────┐                             ┌────┴────┐
   │ Episode │                             │  Ready  │
   │  Loop   │                             │         │
   └────┬────┘                             └────┬────┘
        │                                       │
        │─── episode_start (msgpack) ──────────►│
        │                                       │
   ┌────┴────┐                             ┌────┴────┐
   │  Step   │                             │  Step   │
   │  Loop   │                             │  Loop   │
   └────┬────┘                             └────┬────┘
        │                                       │
        │─── observation (msgpack) ────────────►│
        │                                       │
        │◄────── action (msgpack) ─────────────│
        │                                       │
        │       [repeat until done]             │
        │                                       │
        │─── observation (done=True) ──────────►│
        │                                       │
   └────┴────┘                             └────┴────┘
        │                                       │
        │─── evaluation_complete ──────────────►│
        │                                       │
        │──────── Connection Close ────────────►│
```

## Message Types

### Protocol Negotiation (v1.1)

These messages are exchanged immediately after WebSocket connection to negotiate observation mode and action format.

#### Server Hello (Server → Client)

Sent immediately after connection. Advertises server capabilities.

```python
{
    "type": "server_hello",
    "protocol_version": "1.1",
    "server_type": str,               # "cma" | "seq2seq" | "waypoint"
    "capabilities": {
        "observation_mode": str,      # "egocentric" | "panoramic"
        "action_type": str,           # "discrete" | "waypoint"
        "num_panos": int | None,      # Number of panoramic views (None for egocentric)
        "rgb_shape": List[int],       # Expected RGB shape
        "depth_shape": List[int],     # Expected depth shape
        "action_space": {
            "type": str,              # "discrete" | "continuous"
            "num_actions": int | None,# For discrete: number of actions
            "actions": List[str],     # Action names
        }
    }
}
```

#### Client Hello (Client → Server)

Acknowledges server and confirms configuration.

```python
{
    "type": "client_hello",
    "protocol_version": "1.1",
    "client_type": str,               # "vlnce_eval"
    "configuration": {
        "observation_mode": str,      # "egocentric" | "panoramic"
        "num_panos": int | None,      # Number of pano views (None for egocentric)
    },
    "compatible": bool,               # True if client can satisfy server requirements
}
```

#### Handshake Complete (Server → Client)

Confirms handshake success or failure.

```python
{
    "type": "handshake_complete",
    "status": str,                    # "ok" | "error"
    "message": str | None,            # Error description if status="error"
}
```

#### Handshake Requirements

- Handshake is **mandatory** - both server and client must complete the full handshake before evaluation.
- Server will reject any messages before receiving `client_hello`.
- Client will fail if server does not send `server_hello` within 5 seconds.
- The `--panoramic` flag is optional - client auto-configures from server capabilities.

### 1. Episode Start (Client → Server)

Sent at the beginning of each episode.

```python
{
    "type": "episode_start",
    "episode_id": str,              # Unique episode identifier
    "instruction": {
        "text": str,                # Natural language instruction
        "tokens": List[int] | None, # Tokenized instruction (optional)
        "trajectory_id": str,       # Trajectory identifier
    }
}
```

### 2. Observation (Client → Server)

Sent at each timestep. The server should respond with an action (unless `done=True`).

```python
{
    "type": "observation",
    "episode_id": str,              # Episode identifier
    "step": int,                    # Current step number (0-indexed)
    "rgb": np.ndarray,              # (256, 256, 3) uint8 - RGB image
    "depth": np.ndarray,            # (256, 256, 1) float32 - Depth image
    "instruction": {
        "text": str,                # Natural language instruction
        "tokens": List[int] | None, # Tokenized instruction
        "trajectory_id": str,
    },
    "done": bool,                   # True if episode ended
}
```

### 3. Action (Server → Client)

Response to an observation message.

**Discrete action (CMA, Seq2Seq):**
```python
{
    "type": "action",
    "action": int,                  # Action index (0-5)
}
```

**Waypoint action (HPN, WPN):**
```python
{
    "type": "action",
    "action": {
        "action": "GO_TOWARD_POINT",  # or "STOP"
        "action_args": {              # Only for GO_TOWARD_POINT
            "r": float,               # Distance in meters
            "theta": float,           # Heading in radians
        }
    }
}
```

### 4. Evaluation Complete (Client → Server)

Sent after all episodes are completed.

```python
{
    "type": "evaluation_complete",
    "total_episodes": int,          # Number of episodes evaluated
    "aggregated_metrics": {
        "success": float,           # Success rate (0-1)
        "spl": float,               # SPL score (0-1)
        "ndtw": float,              # nDTW score (0-1)
        "distance_to_goal": float,  # Average distance to goal (meters)
        "path_length": float,       # Average path length (meters)
        "oracle_success": float,    # Oracle success rate (0-1)
        "steps_taken": float,       # Average steps per episode
    }
}
```

## Action Space

### Discrete Actions (CMA, Seq2Seq models)

VLN-CE uses 6 discrete actions:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | STOP | End the episode |
| 1 | MOVE_FORWARD | Move forward 0.25m |
| 2 | TURN_LEFT | Turn left 15 degrees |
| 3 | TURN_RIGHT | Turn right 15 degrees |
| 4 | LOOK_UP | Look up 15 degrees |
| 5 | LOOK_DOWN | Look down 15 degrees |

### Waypoint Actions (HPN, WPN models)

Waypoint models use continuous navigation commands:

| Action | Format | Description |
|--------|--------|-------------|
| STOP | `{"action": "STOP"}` | End the episode |
| GO_TOWARD_POINT | `{"action": "GO_TOWARD_POINT", "action_args": {"r": float, "theta": float}}` | Navigate to waypoint |

**GO_TOWARD_POINT parameters:**
- `r`: Distance in meters to the target waypoint
- `theta`: Heading angle in radians (0 = forward, positive = left)

## Observation Space

### Egocentric Mode (Default - CMA, Seq2Seq models)

#### RGB Image
- Shape: `(256, 256, 3)`
- Dtype: `uint8`
- Range: `[0, 255]`
- Field of view: 90 degrees
- Egocentric first-person view

#### Depth Image
- Shape: `(256, 256, 1)`
- Dtype: `float32`
- Range: `[0, 10]` meters (clipped)
- Field of view: 90 degrees
- Egocentric depth map

### Panoramic Mode (--panoramic flag - HPN, WPN models)

When using waypoint models, observations are captured from 12 viewpoints (30° apart) to provide a 360° panoramic view.

#### Panoramic RGB
- Shape: `(12, 224, 224, 3)`
- Dtype: `uint8`
- Range: `[0, 255]`
- 12 views × 90° FOV each = 360° panorama

#### Panoramic Depth
- Shape: `(12, 256, 256, 1)`
- Dtype: `float32`
- Range: `[0, 10]` meters
- 12 depth maps corresponding to RGB views

#### View Ordering
Views are ordered by heading angle:
- View 0: 0° (forward)
- View 1: 30°
- View 2: 60°
- ...
- View 11: 330°

### Instruction
- Text: Natural language navigation instruction
- Tokens: Pre-tokenized instruction (optional, for BERT-based models)
- Example: "Walk past the table and turn left at the doorway. Enter the bedroom and stop near the bed."

## Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Success (SR)** | Episode success if final position within 3m of goal | [0, 1] |
| **SPL** | Success weighted by Path Length efficiency | [0, 1] |
| **nDTW** | Normalized Dynamic Time Warping - trajectory similarity | [0, 1] |
| **Distance to Goal** | Distance from final position to goal | meters |
| **Path Length** | Total distance traveled | meters |
| **Oracle Success (OS)** | Success if agent ever came within 3m of goal | [0, 1] |
| **Steps Taken** | Number of actions executed | integer |

## WebSocket Configuration

| Setting | Value |
|---------|-------|
| Max message size | 100 MB |
| Action timeout | 300 seconds |
| Protocol | msgpack with numpy support |

## Example Python Server

```python
import asyncio
import msgpack
import msgpack_numpy
import websockets

msgpack_numpy.patch()

class PolicyServer:
    async def handle_client(self, websocket):
        async for message in websocket:
            msg = msgpack.unpackb(message, raw=False)

            if msg["type"] == "episode_start":
                # Initialize episode state
                print(f"Episode started: {msg['episode_id']}")

            elif msg["type"] == "observation":
                if not msg["done"]:
                    # Get observation data
                    rgb = msg["rgb"]        # (256, 256, 3)
                    depth = msg["depth"]    # (256, 256, 1)
                    instruction = msg["instruction"]["text"]

                    # Compute action from your policy
                    action = your_policy(rgb, depth, instruction)

                    # Send action response
                    response = {"type": "action", "action": int(action)}
                    await websocket.send(msgpack.packb(response, use_bin_type=True))

            elif msg["type"] == "evaluation_complete":
                print(f"Evaluation complete: {msg['aggregated_metrics']}")

    async def run(self, host="0.0.0.0", port=8765):
        async with websockets.serve(
            self.handle_client, host, port,
            max_size=100 * 1024 * 1024
        ):
            await asyncio.Future()

if __name__ == "__main__":
    server = PolicyServer()
    asyncio.run(server.run())
```

## Error Handling

### Connection Errors
- Client retries connection on initial failure
- No automatic reconnection after connection loss

### Action Timeout
- Default timeout: 300 seconds
- If server doesn't respond, client raises `TimeoutError`

### Invalid Actions
- Discrete actions outside [0, 5] raise `ValueError`
- Waypoint actions missing `r` or `theta` raise `ValueError`
- Server should always return valid action formats
