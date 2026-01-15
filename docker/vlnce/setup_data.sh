#!/bin/bash
# =============================================================================
# VLN-CE Data Setup Script
# =============================================================================
# Downloads and sets up data for VLN-CE evaluation client:
#   - RxR dataset (optional, for multilingual experiments)
#   - RxR BERT text features (optional, 142GB)
#   - DDPPO depth encoder models
#   - Pretrained model checkpoints (optional)
#
# Note: Matterport3D scenes must be downloaded manually.
#       See https://niessner.github.io/Matterport/
#
# Usage:
#   ./setup_data.sh [OPTIONS]
#
# Options:
#   --all           Download everything
#   --rxr           Download RxR dataset only
#   --rxr-bert      Download RxR BERT features (142GB, requires gsutil)
#   --ddppo         Download DDPPO depth encoder models (672MB)
#   --checkpoints   Download pretrained checkpoints only
#   --help          Show this help message
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DATA_DIR="$PROJECT_ROOT/data"

# Google Drive file IDs
GDRIVE_RXR_DATASET="145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t"
GDRIVE_CMA_EN="1fe0-w6ElGwX5VWtESKSM_20VY7sfn4fV"
GDRIVE_CMA_HI="1z84xMJ1LP2NO_jpJjFdymejXQqhU6zZH"
GDRIVE_CMA_TE="13mGjoKyJaWSJsnoQ-el4oIAlai0l7zfQ"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

download_gdrive() {
    local file_id="$1"
    local output_path="$2"
    local description="$3"

    if [[ -f "$output_path" ]]; then
        print_warning "$description already exists, skipping download."
        return 0
    fi

    print_info "Downloading $description..."
    gdown "https://drive.google.com/uc?id=$file_id" -O "$output_path"

    if [[ -f "$output_path" ]]; then
        print_success "Downloaded $description"
    else
        print_error "Failed to download $description"
        return 1
    fi
}

# =============================================================================
# Download Functions
# =============================================================================

download_rxr_dataset() {
    print_header "RxR Dataset Download"

    check_command gdown || return 1

    local rxr_dir="$DATA_DIR/datasets/RxR_VLNCE_v0"
    local rxr_zip="$DATA_DIR/datasets/RxR_VLNCE_v0.zip"

    if [[ -d "$rxr_dir" ]]; then
        print_warning "RxR dataset already exists at $rxr_dir"
        read -p "Re-download? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    mkdir -p "$DATA_DIR/datasets"

    download_gdrive "$GDRIVE_RXR_DATASET" "$rxr_zip" "RxR_VLNCE_v0.zip"

    if [[ -f "$rxr_zip" ]]; then
        print_info "Extracting RxR dataset..."
        unzip -o "$rxr_zip" -d "$DATA_DIR/datasets/"
        print_success "Extracted RxR dataset to $rxr_dir"
        rm -f "$rxr_zip"
    fi
}

download_rxr_bert_features() {
    print_header "RxR BERT Text Features Download"

    echo "This will download ~142GB of BERT text features using gsutil."
    echo "Features will be saved to: $DATA_DIR/datasets/RxR_VLNCE_v0/text_features/"
    echo ""

    check_command gsutil || {
        print_error "gsutil is required. Install Google Cloud SDK:"
        echo "  https://cloud.google.com/sdk/docs/install"
        return 1
    }

    read -p "Proceed with download? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_warning "Skipping BERT features download."
        return 0
    fi

    local bert_dir="$DATA_DIR/datasets/RxR_VLNCE_v0/text_features"
    mkdir -p "$bert_dir"

    print_info "Downloading BERT text features (this may take a while)..."
    gsutil -m cp -R gs://rxr-data/text_features/* "$bert_dir/"

    print_success "Downloaded BERT text features"
}

download_ddppo() {
    print_header "DDPPO Depth Encoder Models Download"

    local ddppo_dir="$DATA_DIR/ddppo-models"
    local ddppo_file="$ddppo_dir/gibson-2plus-resnet50.pth"

    if [[ -f "$ddppo_file" ]]; then
        print_warning "DDPPO models already exist at $ddppo_dir"
        read -p "Re-download? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    mkdir -p "$ddppo_dir"

    print_info "Downloading ddppo-models.zip (672MB)..."
    local zip_file="$DATA_DIR/ddppo-models.zip"
    wget -q --show-progress https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip -O "$zip_file"

    if [[ -f "$zip_file" ]]; then
        print_info "Extracting DDPPO models..."
        # Zip contains data/ddppo-models/, so extract to temp and move
        local temp_dir="$DATA_DIR/ddppo_temp"
        mkdir -p "$temp_dir"
        unzip -q "$zip_file" -d "$temp_dir"

        # Move files to correct location
        mv "$temp_dir"/data/ddppo-models/* "$ddppo_dir/"
        rm -rf "$temp_dir"
        rm -f "$zip_file"
        print_success "Extracted DDPPO models to $ddppo_dir"
    else
        print_error "Failed to download DDPPO models"
        return 1
    fi
}

download_checkpoints() {
    print_header "Pretrained Model Checkpoints Download"

    check_command gdown || return 1

    local ckpt_dir="$DATA_DIR/checkpoints"
    mkdir -p "$ckpt_dir"

    echo "Available pretrained models:"
    echo "  1. CMA English (rxr_cma_en) - 196MB"
    echo "  2. CMA Hindi (rxr_cma_hi) - 196MB"
    echo "  3. CMA Telugu (rxr_cma_te) - 196MB"
    echo "  4. All of the above"
    echo ""

    read -p "Select option (1-4, or 0 to skip): " choice

    case $choice in
        1)
            download_gdrive "$GDRIVE_CMA_EN" "$ckpt_dir/rxr_cma_en.pth" "CMA English checkpoint"
            ;;
        2)
            download_gdrive "$GDRIVE_CMA_HI" "$ckpt_dir/rxr_cma_hi.pth" "CMA Hindi checkpoint"
            ;;
        3)
            download_gdrive "$GDRIVE_CMA_TE" "$ckpt_dir/rxr_cma_te.pth" "CMA Telugu checkpoint"
            ;;
        4)
            download_gdrive "$GDRIVE_CMA_EN" "$ckpt_dir/rxr_cma_en.pth" "CMA English checkpoint"
            download_gdrive "$GDRIVE_CMA_HI" "$ckpt_dir/rxr_cma_hi.pth" "CMA Hindi checkpoint"
            download_gdrive "$GDRIVE_CMA_TE" "$ckpt_dir/rxr_cma_te.pth" "CMA Telugu checkpoint"
            ;;
        0)
            print_warning "Skipping checkpoint download."
            ;;
        *)
            print_error "Invalid option"
            return 1
            ;;
    esac
}

show_status() {
    print_header "Current Data Status"

    echo "Checking data directories..."
    echo ""

    # R2R Dataset
    local r2r_dir="$DATA_DIR/datasets/R2R_VLNCE_v1-3_preprocessed"
    if [[ -d "$r2r_dir" ]]; then
        print_success "R2R Dataset: Found at $r2r_dir"
    else
        print_warning "R2R Dataset: NOT FOUND"
    fi

    # RxR Dataset
    local rxr_dir="$DATA_DIR/datasets/RxR_VLNCE_v0"
    if [[ -d "$rxr_dir" ]]; then
        print_success "RxR Dataset: Found at $rxr_dir"
    else
        print_warning "RxR Dataset: NOT FOUND"
    fi

    # MP3D Scenes (manual download required)
    local mp3d_count=$(find "$DATA_DIR/scene_datasets/mp3d" -name "*.glb" 2>/dev/null | wc -l)
    if [[ $mp3d_count -gt 0 ]]; then
        print_success "MP3D Scenes: Found $mp3d_count .glb files"
    else
        print_warning "MP3D Scenes: NOT FOUND (manual download required)"
        print_info "  See https://niessner.github.io/Matterport/"
    fi

    # DDPPO Models
    local ddppo_count=$(find "$DATA_DIR/ddppo-models" -name "*.pth" 2>/dev/null | wc -l)
    if [[ $ddppo_count -gt 0 ]]; then
        print_success "DDPPO Models: Found $ddppo_count model files"
    else
        print_warning "DDPPO Models: NOT FOUND"
    fi

    # Checkpoints
    local ckpt_count=$(find "$DATA_DIR/checkpoints" -name "*.pth" 2>/dev/null | wc -l)
    if [[ $ckpt_count -gt 0 ]]; then
        print_success "Checkpoints: Found $ckpt_count checkpoint files"
    else
        print_warning "Checkpoints: NOT FOUND (optional, for evaluation)"
    fi

    # RxR BERT Features
    local bert_dir="$DATA_DIR/datasets/RxR_VLNCE_v0/text_features"
    if [[ -d "$bert_dir" ]] && [[ "$(ls -A $bert_dir 2>/dev/null)" ]]; then
        print_success "RxR BERT Features: Found at $bert_dir"
    else
        print_warning "RxR BERT Features: NOT FOUND (optional, for RxR experiments)"
    fi

    echo ""
}

show_help() {
    echo "VLN-CE Data Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all           Download everything"
    echo "  --rxr           Download RxR dataset only"
    echo "  --rxr-bert      Download RxR BERT features (142GB, requires gsutil)"
    echo "  --ddppo         Download DDPPO depth encoder models (672MB)"
    echo "  --checkpoints   Download pretrained checkpoints only"
    echo "  --status        Show current data status"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --status                    # Check what's installed"
    echo "  $0 --ddppo                     # Download DDPPO depth encoder models"
    echo "  $0 --rxr --checkpoints         # Download RxR and checkpoints"
    echo "  $0 --all                       # Download everything"
    echo ""
    echo "Note: Matterport3D scenes must be downloaded manually."
    echo "      See https://niessner.github.io/Matterport/"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "VLN-CE Data Setup"

    echo "Project root: $PROJECT_ROOT"
    echo "Data directory: $DATA_DIR"
    echo ""
    echo "Note: Matterport3D scenes must be downloaded manually."
    echo "      See https://niessner.github.io/Matterport/"
    echo ""

    # Parse arguments
    local do_rxr=false
    local do_rxr_bert=false
    local do_ddppo=false
    local do_checkpoints=false
    local show_status_only=false

    # Default to download all if no arguments provided
    if [[ $# -eq 0 ]]; then
        do_rxr=true
        do_ddppo=true
        do_checkpoints=true
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                do_rxr=true
                do_ddppo=true
                do_checkpoints=true
                shift
                ;;
            --rxr)
                do_rxr=true
                shift
                ;;
            --rxr-bert)
                do_rxr_bert=true
                shift
                ;;
            --ddppo)
                do_ddppo=true
                shift
                ;;
            --checkpoints)
                do_checkpoints=true
                shift
                ;;
            --status)
                show_status_only=true
                shift
                ;;
            --help|-h)
                show_help
                return 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                return 1
                ;;
        esac
    done

    if [[ "$show_status_only" == true ]]; then
        show_status
        return 0
    fi

    # Check gdown is available
    if ! check_command gdown; then
        print_info "Installing gdown..."
        pip install gdown
    fi

    # Execute downloads
    if [[ "$do_rxr" == true ]]; then
        download_rxr_dataset
    fi

    if [[ "$do_rxr_bert" == true ]]; then
        download_rxr_bert_features
    fi

    if [[ "$do_ddppo" == true ]]; then
        download_ddppo
    fi

    if [[ "$do_checkpoints" == true ]]; then
        download_checkpoints
    fi

    # Show final status
    show_status

    print_header "Setup Complete"
    echo "Next steps:"
    echo "  1. Download Matterport3D scenes manually"
    echo "  2. Build Docker container: docker build -t vlnce-client ."
    echo "  3. Run evaluation with docker compose"
    echo ""
}

main "$@"
