#!/bin/bash
# Setup script for VLN-CE Policy Server data
#
# Downloads checkpoints, vocabulary, and DDPPO models to the correct locations
# for use with the Docker policy server.
#
# Usage:
#   cd docker/policy_server
#   ./setup_data.sh           # Download all
#   ./setup_data.sh cma       # Download CMA checkpoint only
#   ./setup_data.sh seq2seq   # Download Seq2Seq checkpoint only
#   ./setup_data.sh waypoint  # Download HPN/WPN checkpoints only
#   ./setup_data.sh vocab     # Download vocabulary only
#   ./setup_data.sh ddppo     # Download DDPPO models only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check for required tools
check_dependencies() {
    if ! command -v gdown &> /dev/null; then
        warn "gdown not found. Installing via pip..."
        pip install gdown
    fi

    if ! command -v wget &> /dev/null; then
        error "wget is required but not installed."
    fi

    if ! command -v unzip &> /dev/null; then
        error "unzip is required but not installed."
    fi
}

# Create directory structure
create_dirs() {
    info "Creating data directories..."
    mkdir -p "${DATA_DIR}/checkpoints"
    mkdir -p "${DATA_DIR}/datasets"
    mkdir -p "${DATA_DIR}/ddppo-models"
}

# Download CMA checkpoint
download_cma() {
    local ckpt_path="${DATA_DIR}/checkpoints/CMA_PM_DA_Aug.pth"
    if [ -f "$ckpt_path" ]; then
        warn "CMA checkpoint already exists: $ckpt_path"
        return
    fi
    info "Downloading CMA_PM_DA_Aug.pth (141MB)..."
    gdown https://drive.google.com/uc?id=1o9PgBT38BH9pw_7V1QB3XUkY8auJqGKw -O "$ckpt_path"
    info "CMA checkpoint downloaded: $ckpt_path"
}

# Download Seq2Seq checkpoint
download_seq2seq() {
    local ckpt_path="${DATA_DIR}/checkpoints/Seq2Seq_DA.pth"
    if [ -f "$ckpt_path" ]; then
        warn "Seq2Seq checkpoint already exists: $ckpt_path"
        return
    fi
    info "Downloading Seq2Seq_DA.pth (135MB)..."
    gdown https://drive.google.com/uc?id=12swcou9g5jwR31GbQU1wJ88E_8j--Qi5 -O "$ckpt_path"
    info "Seq2Seq checkpoint downloaded: $ckpt_path"
}

# Download Waypoint checkpoints (HPN and WPN)
download_waypoint() {
    local hpn_path="${DATA_DIR}/checkpoints/HPN.pth"
    local wpn_path="${DATA_DIR}/checkpoints/WPN.pth"

    if [ -f "$hpn_path" ]; then
        warn "HPN checkpoint already exists: $hpn_path"
    else
        info "Downloading HPN.pth (97MB)..."
        gdown https://drive.google.com/uc?id=1W_q1cqP7g6Y6jHaXKKyFDLpaE3pdRJnI -O "$hpn_path"
        info "HPN checkpoint downloaded: $hpn_path"
    fi

    if [ -f "$wpn_path" ]; then
        warn "WPN checkpoint already exists: $wpn_path"
    else
        info "Downloading WPN.pth (97MB)..."
        gdown https://drive.google.com/uc?id=1XaJbkPYsVZGoM2pyJJ9umeuQ1u8kl9Fm -O "$wpn_path"
        info "WPN checkpoint downloaded: $wpn_path"
    fi
}

# Download vocabulary (R2R dataset)
download_vocab() {
    local vocab_dir="${DATA_DIR}/datasets/R2R_VLNCE_v1-3_preprocessed"
    local vocab_file="${vocab_dir}/train/train.json.gz"

    if [ -f "$vocab_file" ]; then
        warn "Vocabulary already exists: $vocab_file"
        return
    fi

    info "Downloading R2R_VLNCE_v1-3_preprocessed.zip (250MB)..."
    local zip_file="${DATA_DIR}/datasets/R2R_VLNCE_v1-3_preprocessed.zip"
    gdown https://drive.google.com/uc?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr -O "$zip_file"

    info "Extracting vocabulary..."
    unzip -q "$zip_file" -d "${DATA_DIR}/datasets/"
    rm "$zip_file"
    info "Vocabulary extracted: $vocab_dir"
}

# Download DDPPO depth encoder models
download_ddppo() {
    local ddppo_dir="${DATA_DIR}/ddppo-models"
    local ddppo_file="${ddppo_dir}/gibson-2plus-resnet50.pth"

    if [ -f "$ddppo_file" ]; then
        warn "DDPPO models already exist: $ddppo_file"
        return
    fi

    info "Downloading ddppo-models.zip (672MB)..."
    local zip_file="${DATA_DIR}/ddppo-models.zip"
    wget -q --show-progress https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip -O "$zip_file"

    info "Extracting DDPPO models..."
    # Zip contains data/ddppo-models/, so extract to temp and move
    local temp_dir="${DATA_DIR}/ddppo_temp"
    mkdir -p "$temp_dir"
    unzip -q "$zip_file" -d "$temp_dir"

    # Move files to correct location
    mkdir -p "$ddppo_dir"
    mv "$temp_dir"/data/ddppo-models/* "$ddppo_dir/"
    rm -rf "$temp_dir"
    rm "$zip_file"
    info "DDPPO models extracted: $ddppo_dir/"
}

# Download all data
download_all() {
    download_vocab
    download_ddppo
    download_cma
    download_seq2seq
    download_waypoint
}

# Print usage
print_usage() {
    echo "VLN-CE Policy Server Data Setup"
    echo ""
    echo "Usage: $0 [component]"
    echo ""
    echo "Components:"
    echo "  all       Download everything (default)"
    echo "  cma       Download CMA checkpoint only"
    echo "  seq2seq   Download Seq2Seq checkpoint only"
    echo "  waypoint  Download HPN and WPN checkpoints"
    echo "  vocab     Download vocabulary (R2R dataset)"
    echo "  ddppo     Download DDPPO depth encoder models"
    echo ""
    echo "Data will be downloaded to: ${DATA_DIR}"
    echo ""
    echo "Expected structure after download:"
    echo "  data/"
    echo "  ├── checkpoints/"
    echo "  │   ├── CMA_PM_DA_Aug.pth"
    echo "  │   ├── Seq2Seq_DA.pth"
    echo "  │   ├── HPN.pth"
    echo "  │   └── WPN.pth"
    echo "  ├── datasets/"
    echo "  │   └── R2R_VLNCE_v1-3_preprocessed/"
    echo "  │       └── train/"
    echo "  │           └── train.json.gz"
    echo "  └── ddppo-models/"
    echo "      └── gibson-2plus-resnet50.pth"
}

# Main
main() {
    local component="${1:-all}"

    check_dependencies
    create_dirs

    case "$component" in
        all)
            info "Downloading all data..."
            download_all
            ;;
        cma)
            download_cma
            ;;
        seq2seq)
            download_seq2seq
            ;;
        waypoint)
            download_waypoint
            ;;
        vocab)
            download_vocab
            ;;
        ddppo)
            download_ddppo
            ;;
        -h|--help|help)
            print_usage
            exit 0
            ;;
        *)
            error "Unknown component: $component"
            print_usage
            exit 1
            ;;
    esac

    echo ""
    info "Setup complete!"
    echo ""
    echo "To run the policy server:"
    echo "  docker compose --profile cma up"
    echo ""
}

main "$@"
