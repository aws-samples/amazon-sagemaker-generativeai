#!/bin/bash
set -euo pipefail

# SAMA Science Agents - Setup Script
# Removes Q CLI, installs Kiro CLI, installs MCP server dependencies,
# and configures the .kiro directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Step 1: Remove Q CLI
log_info "Removing Q CLI (if installed)..."

if command -v q &> /dev/null; then
    Q_PATH="$(which q)"
    log_info "Found Q CLI at $Q_PATH"
    sudo rm -f "$Q_PATH" 2>/dev/null || rm -f "$Q_PATH" 2>/dev/null || log_warn "Could not remove $Q_PATH — try manually: sudo rm -f $Q_PATH"
    rm -rf ~/.local/share/amazon-q 2>/dev/null || true
    rm -rf ~/.config/amazon-q 2>/dev/null || true
    rm -rf ~/.q 2>/dev/null || true
    log_success "Q CLI removed"
else
    log_info "Q CLI not found, skipping removal"
fi

# Step 2: Install Kiro CLI
log_info "Installing Kiro CLI..."

# Ensure ~/.local/bin is on PATH (where kiro-cli installs to)
if ! grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    log_info "Added ~/.local/bin to PATH in ~/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"

if command -v kiro-cli &> /dev/null; then
    log_info "Kiro CLI already installed: $(kiro-cli --version 2>/dev/null || echo 'unknown version')"
else
    curl -fsSL https://cli.kiro.dev/install | bash
    log_success "Kiro CLI installed"
fi

# Step 3: Verify Python
log_info "Checking Python at $PYTHON..."

if [[ ! -x "$PYTHON" ]]; then
    log_warn "$PYTHON not found, falling back to system python3"
    PYTHON="$(which python3)"
fi

if [[ -z "$PYTHON" ]] || [[ ! -x "$PYTHON" ]]; then
    log_error "No python3 found. Set PYTHON env var and rerun: PYTHON=/path/to/python3 bash install.sh"
    exit 1
fi

log_success "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# Step 4: Install Python dependencies
log_info "Installing Python dependencies..."

$PYTHON -m pip install -q -r "$SCRIPT_DIR/requirements.txt"

for server_dir in "$SCRIPT_DIR"/MCP_servers/sama-*/; do
    if [[ -f "$server_dir/pyproject.toml" ]]; then
        server_name="$(basename "$server_dir")"
        log_info "Installing $server_name..."
        $PYTHON -m pip install -q -e "$server_dir"
    fi
done


echo ""
log_success "Setup complete!"
echo ""
echo "  To start:  cd $SCRIPT_DIR && kiro-cli chat --trust-all-tools"
echo "  Skills:    #supervised-finetuning  or  #grpo-rlvr"
echo "  Check:     /mcp  (should show 6 green servers)"
echo ""
