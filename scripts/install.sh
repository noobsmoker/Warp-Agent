#!/bin/bash
# ============================================================================
# Warp Agent Installer
# ============================================================================
# One-line installer for the Warp-Cortex-enabled Hermes Agent (Warp Agent).
# Works on Linux, macOS, WSL2, and Android (Termux).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/noobsmoker/warp-agent/main/scripts/install.sh | bash
#
# This script:
# 1. Detects platform (desktop/server vs Android/Termux)
# 2. Creates Python 3.11 virtual environment
# 3. Installs all dependencies including Warp-Cortex components
# 4. Sets up warp-claw command
# 5. Optionally runs setup wizard
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="3.11"

is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

get_command_link_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo "$PREFIX/bin"
    else
        echo "$HOME/.local/bin"
    fi
}

get_command_link_display_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo '$PREFIX/bin'
    else
        echo '~/.local/bin'
    fi
}

detect_platform() {
    if is_termux; then
        echo "termux"
    else
        case "$(uname -s)" in
            Linux*)     echo "linux" ;;
            Darwin*)    echo "macos" ;;
            CYGWIN*)    echo "cygwin" ;;
            MINGW*)     echo "mingw" ;;
            *)          echo "unknown" ;;
        esac
    fi
}

install_uv() {
    echo -e "${CYAN}Installing uv package manager...${NC}"
    if command -v uv >/dev/null 2>&1; then
        echo "uv already installed."
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

setup_venv() {
    local venv_path="$HOME/.warp-agent/venv"

    echo -e "${CYAN}Setting up Python $PYTHON_VERSION virtual environment at $venv_path...${NC}"
    
    if [ ! -d "$venv_path" ]; then
        uv venv "$venv_path" --python "$PYTHON_VERSION"
    else
        echo "Virtual environment already exists."
    fi
    
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source "$venv_path/bin/activate"
    
    echo -e "${CYAN}Installing Warp-Claw with all dependencies...${NC}"
    uv pip install -e ".[all]"
    
    # Verify installation
    if ! python -c "import warp_cortex; print('Warp-Cortex imported successfully')" >/dev/null 2>&1; then
        echo -e "${RED}Error: Warp-Cortex module not found. Installation may have failed.${NC}"
        exit 1
    fi
}

create_symlinks() {
    local bin_dir="$(get_command_link_dir)"
    local venv_path="$HOME/.warp-agent/venv"

    echo -e "${CYAN}Creating symlink for warp-agent command in $bin_dir...${NC}"

    mkdir -p "$bin_dir"
    ln -sf "$venv_path/bin/warp-agent" "$bin_dir/warp-agent"
    
    # Add to PATH if needed
    if ! echo "$PATH" | grep -q "$bin_dir"; then
        echo -e "${YELLOW}Adding $bin_dir to PATH. You may need to restart your shell.${NC}"
        export PATH="$bin_dir:$PATH"
    fi
}

run_setup() {
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo -e "${CYAN}To start using Warp Agent:${NC}"
    echo "  warp-agent             # Interactive CLI"
    echo "  warp-agent --tui       # Modern TUI"
    echo "  warp-agent gateway     # Start messaging gateway"
    echo ""
    echo -e "${CYAN}First-time setup:${NC}"
    echo "  warp-agent setup       # Configure providers and tools"
    echo ""
    echo -e "${YELLOW}Note: Restart your shell if the 'warp-agent' command is not found.${NC}"
}

main() {
    local platform="$(detect_platform)"
    echo -e "${GREEN}Warp-Claw Installer${NC}"
    echo -e "${CYAN}Platform detected: $platform${NC}"
    echo ""
    
    install_uv
    setup_venv
    create_symlinks
    run_setup
}

main "$@"