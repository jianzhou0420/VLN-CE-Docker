#!/bin/bash
# Unified entrypoint for VLN-CE Policy Server
#
# Usage: entrypoint.sh <server_type> [args...]
#   server_type: cma, seq2seq, waypoint
#   args: passed to the selected server
#
# Examples:
#   entrypoint.sh cma --port 8765
#   entrypoint.sh seq2seq --port 8765 --checkpoint /app/data/checkpoints/my_model.pth
#   entrypoint.sh waypoint --port 8765 --num-panos 12

set -e

SERVER_TYPE="${1:-help}"
shift 2>/dev/null || true

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate vlnce

case "$SERVER_TYPE" in
    cma)
        exec python -m vlnce_server.cma_server "$@"
        ;;
    seq2seq)
        exec python -m vlnce_server.seq2seq_server "$@"
        ;;
    waypoint)
        exec python -m vlnce_server.waypoint_server "$@"
        ;;
    --help|-h|help)
        echo "VLN-CE Policy Server"
        echo ""
        echo "Usage: docker run vlnce-policy-server <server_type> [options]"
        echo ""
        echo "Server types:"
        echo "  cma       - CMA (Cross-Modal Attention) policy server"
        echo "              Egocentric observations, discrete actions (0-5)"
        echo ""
        echo "  seq2seq   - Seq2Seq policy server"
        echo "              Egocentric observations, discrete actions (0-5)"
        echo ""
        echo "  waypoint  - Waypoint (HPN/WPN) policy server"
        echo "              Panoramic observations (12 views), waypoint actions"
        echo ""
        echo "Common options:"
        echo "  --host HOST          Server host (default: 0.0.0.0)"
        echo "  --port PORT          Server port (default: 8765)"
        echo "  --checkpoint PATH    Model checkpoint path"
        echo "  --vocab PATH         Vocabulary file path"
        echo "  --ddppo-checkpoint   DDPPO depth encoder weights"
        echo "  --config PATH        Config YAML file"
        echo "  --gpu GPU_ID         GPU device ID (default: 0)"
        echo "  -v, --verbose        Enable verbose logging"
        echo ""
        echo "Waypoint-specific options:"
        echo "  --num-panos N        Number of panoramic views (default: 12)"
        echo ""
        echo "Example:"
        echo "  docker run --rm -it --gpus all --network=host \\"
        echo "    -v ./data/checkpoints:/app/data/checkpoints:ro \\"
        echo "    -v ./data/datasets:/app/data/datasets:ro \\"
        echo "    -v ./data/ddppo-models:/app/data/ddppo-models:ro \\"
        echo "    vlnce-policy-server cma --port 8765"
        echo ""
        echo "Volume mounts required:"
        echo "  /app/data/checkpoints  - Model checkpoint files (.pth)"
        echo "  /app/data/datasets     - Vocabulary files (train.json.gz)"
        echo "  /app/data/ddppo-models - DDPPO depth encoder weights"
        exit 0
        ;;
    *)
        echo "Error: Unknown server type '$SERVER_TYPE'"
        echo ""
        echo "Valid server types: cma, seq2seq, waypoint"
        echo "Run with --help for usage information"
        exit 1
        ;;
esac
