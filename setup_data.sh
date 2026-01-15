#!/bin/bash
# =============================================================================
# VLN-CE Data Setup - Master Script
# =============================================================================
# Launches setup_data.sh for both evaluation client and policy server.
#
# Usage:
#   ./setup_data.sh                     # Run both (interactive menu)
#   ./setup_data.sh --vlnce [OPTIONS]   # Run vlnce client setup only
#   ./setup_data.sh --policy [OPTIONS]  # Run policy server setup only
#   ./setup_data.sh --all               # Run both with default options
#   ./setup_data.sh --help              # Show this help
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLNCE_SETUP="${SCRIPT_DIR}/docker/vlnce/setup_data.sh"
POLICY_SETUP="${SCRIPT_DIR}/docker/policy_server/setup_data.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "VLN-CE Data Setup - Master Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --vlnce [ARGS]    Run evaluation client setup only"
    echo "                    ARGS are passed to docker/vlnce/setup_data.sh"
    echo "                    Available: --rxr, --rxr-bert, --ddppo, --checkpoints, --all"
    echo ""
    echo "  --policy [ARGS]   Run policy server setup only"
    echo "                    ARGS are passed to docker/policy_server/setup_data.sh"
    echo "                    Available: cma, seq2seq, waypoint, vocab, ddppo, all"
    echo ""
    echo "  --all             Run both setups with default options"
    echo "  --status          Show data status for both components"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Interactive menu"
    echo "  $0 --all                    # Download all data for both"
    echo "  $0 --policy cma             # Download CMA checkpoint only"
    echo "  $0 --status                 # Check data status"
    echo ""
    echo "Note: Matterport3D scenes must be downloaded manually."
    echo "      See https://niessner.github.io/Matterport/"
    echo ""
}

run_vlnce_setup() {
    print_header "Running VLN-CE Evaluation Client Setup"

    if [[ ! -f "$VLNCE_SETUP" ]]; then
        print_error "VLN-CE setup script not found: $VLNCE_SETUP"
        exit 1
    fi

    chmod +x "$VLNCE_SETUP"
    "$VLNCE_SETUP" "$@"
}

run_policy_setup() {
    print_header "Running Policy Server Setup"

    if [[ ! -f "$POLICY_SETUP" ]]; then
        print_error "Policy server setup script not found: $POLICY_SETUP"
        exit 1
    fi

    chmod +x "$POLICY_SETUP"
    "$POLICY_SETUP" "$@"
}

show_status() {
    print_header "VLN-CE Data Status"

    echo "=== Evaluation Client Data ==="
    if [[ -f "$VLNCE_SETUP" ]]; then
        "$VLNCE_SETUP" --status 2>/dev/null || print_warn "Could not get vlnce status"
    fi

    echo ""
    echo "=== Policy Server Data ==="
    local policy_data="${SCRIPT_DIR}/docker/policy_server/data"
    if [[ -d "$policy_data" ]]; then
        echo "Checking ${policy_data}..."

        # Checkpoints
        local ckpt_count=$(find "${policy_data}/checkpoints" -name "*.pth" 2>/dev/null | wc -l)
        if [[ $ckpt_count -gt 0 ]]; then
            print_info "Checkpoints: Found $ckpt_count files"
            ls -lh "${policy_data}/checkpoints/"*.pth 2>/dev/null | awk '{print "    " $NF ": " $5}'
        else
            print_warn "Checkpoints: NOT FOUND"
        fi

        # DDPPO
        if [[ -f "${policy_data}/ddppo-models/gibson-2plus-resnet50.pth" ]]; then
            print_info "DDPPO models: Found"
        else
            print_warn "DDPPO models: NOT FOUND"
        fi

        # Vocab
        if [[ -d "${policy_data}/datasets/R2R_VLNCE_v1-3_preprocessed" ]]; then
            print_info "Vocabulary: Found"
        else
            print_warn "Vocabulary: NOT FOUND"
        fi
    else
        print_warn "Policy server data directory not found: $policy_data"
    fi
}

interactive_menu() {
    print_header "VLN-CE Data Setup"

    echo "This script sets up data for VLN-CE Docker containers."
    echo ""
    echo "Components:"
    echo "  1. Evaluation Client (docker/vlnce)"
    echo "     - RxR dataset, DDPPO models, checkpoints"
    echo ""
    echo "  2. Policy Server (docker/policy_server)"
    echo "     - Model checkpoints (CMA, Seq2Seq, Waypoint)"
    echo "     - Vocabulary, DDPPO models"
    echo ""
    echo "Note: Matterport3D scenes must be downloaded manually."
    echo "      See https://niessner.github.io/Matterport/"
    echo ""
    echo "Select setup to run:"
    echo "  1) Evaluation Client only"
    echo "  2) Policy Server only"
    echo "  3) Both (run sequentially) [default]"
    echo "  4) Show status"
    echo "  0) Exit"
    echo ""

    read -p "Enter choice [0-4, default=3]: " choice
    choice="${choice:-3}"

    case $choice in
        1)
            run_vlnce_setup
            ;;
        2)
            run_policy_setup
            ;;
        3)
            run_vlnce_setup
            run_policy_setup
            ;;
        4)
            show_status
            ;;
        0)
            echo "Exiting."
            exit 0
            ;;
        *)
            print_error "Invalid choice: $choice"
            exit 1
            ;;
    esac
}

# Main
main() {
    if [[ $# -eq 0 ]]; then
        interactive_menu
        exit 0
    fi

    case "$1" in
        --vlnce)
            shift
            run_vlnce_setup "$@"
            ;;
        --policy)
            shift
            run_policy_setup "$@"
            ;;
        --all)
            run_vlnce_setup --all
            run_policy_setup all
            ;;
        --status)
            show_status
            ;;
        --help|-h|help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
