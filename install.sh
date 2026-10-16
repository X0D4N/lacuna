#!/usr/bin/env bash
# Lacuna installer — Linux (Debian-based and Arch-based).
# Usage: ./install.sh
set -euo pipefail

if [ -f /etc/debian_version ]; then
    echo "[lacuna] Debian-based distro detected (apt)"
    sudo apt update
    sudo apt install -y python3 python3-pip
elif [ -f /etc/arch-release ]; then
    echo "[lacuna] Arch-based distro detected (pacman)"
    sudo pacman -Sy --noconfirm python python-pip
else
    echo "[lacuna] Unknown distro; please ensure python3 (>=3.10) and pip are installed, then re-run."
    exit 1
fi

python3 --version
pip install .
echo "[lacuna] installed. Try: lacuna simulate"
