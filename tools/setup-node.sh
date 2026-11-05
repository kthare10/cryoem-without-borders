#!/bin/bash
# setup-node.sh — install the CryoEM pre-processing stack on the GPU node.
#
#   bash ~/tools/setup-node.sh [CUDA_VERSION] [INSTALL_RELION]
#     CUDA_VERSION    e.g. 12.4 (default). MotionCor3 links libnvToolsExt, so stay on 12.x.
#     INSTALL_RELION  true|false (default true) — build RELION 5.0 (CUDA, no GUI).
#
# Run as the login user (NOT via fablib's timeout=, which runs as root). Uses sudo
# only where needed. Idempotent: re-running skips anything already installed.
#
# Exit codes: 0 = ready, 3 = NVIDIA driver installed but a reboot is needed
#             (provision.py reboots and re-runs this script), other = failure.
set -euo pipefail

CUDA_VERSION="${1:-12.4}"
INSTALL_RELION="${2:-true}"
CUDA_PKG="cuda-toolkit-${CUDA_VERSION/./-}"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
SRC="$HOME/src"
CTFFIND_URL="https://grigoriefflab.umassmed.edu/system/tdf?path=ctffind-4.1.14-linux64.tar.gz&file=1&type=node&id=26"

export DEBIAN_FRONTEND=noninteractive
log() { echo "### SETUP: $*"; }
apt_install() { sudo -E apt-get install -y -q "$@" >/dev/null; }

mkdir -p "$SRC" "$HOME/cryoem-work"

# --- 1. base packages -------------------------------------------------------
log "apt base packages"
sudo -E apt-get update -q >/dev/null
apt_install build-essential git wget curl cmake pkg-config ca-certificates \
    libtiff-dev libpng-dev libfftw3-dev mpi-default-bin mpi-default-dev \
    ghostscript python3-venv python3-pip rsync iperf3 "linux-headers-$(uname -r)"

# --- 2. NVIDIA driver + CUDA toolkit (NVIDIA apt repo) -----------------------
if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ] && \
   ! ls /etc/apt/sources.list.d/ | grep -q cuda; then
    log "adding NVIDIA CUDA apt repo"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb >/dev/null
    sudo -E apt-get update -q >/dev/null
fi

if ! nvidia-smi >/dev/null 2>&1; then
    log "installing NVIDIA driver"
    apt_install cuda-drivers
    # Try to load the driver without a reboot (works when nouveau is not bound).
    sudo modprobe -r nouveau 2>/dev/null || true
    sudo modprobe nvidia 2>/dev/null || true
    if ! nvidia-smi >/dev/null 2>&1; then
        echo 'blacklist nouveau' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
        echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
        sudo update-initramfs -u >/dev/null 2>&1 || true
        log "REBOOT_REQUIRED"
        exit 3
    fi
fi
log "driver OK: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | paste -sd';')"

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    log "installing $CUDA_PKG"
    apt_install "$CUDA_PKG"
fi
grep -q "$CUDA_HOME/bin" "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<EOF
export PATH=$CUDA_HOME/bin:/opt/relion/bin:\$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
EOF
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "$CUDA_HOME/lib64" | sudo tee /etc/ld.so.conf.d/cuda-cryoem.conf >/dev/null && sudo ldconfig

# --- 3. MotionCor3 (frame alignment) ----------------------------------------
if ! command -v MotionCor3 >/dev/null 2>&1; then
    log "building MotionCor3 (several minutes)"
    [ -d "$SRC/MotionCor3" ] || git clone -q https://github.com/czimaginginstitute/MotionCor3.git "$SRC/MotionCor3"
    # Ubuntu 22.04 g++ links PIE by default, but the bundled LibSrc/Lib/*.a are not
    # -fPIE ("relocation R_X86_64_32 ... recompile with -fPIE"): link the exe non-PIE.
    grep -q -- '-no-pie' "$SRC/MotionCor3/makefile11" || \
        sed -i 's/@$(NVCC) -g -G -m64 $(OBJS)/@$(NVCC) -g -G -m64 -Xcompiler -no-pie $(OBJS)/' "$SRC/MotionCor3/makefile11"
    ( cd "$SRC/MotionCor3" && make -s exe -f makefile11 CUDAHOME="$CUDA_HOME" -j"$(nproc)" >"$SRC/motioncor3-build.log" 2>&1 ) \
        || { tail -30 "$SRC/motioncor3-build.log"; exit 1; }
    sudo install -m 0755 "$SRC/MotionCor3/MotionCor3" /usr/local/bin/MotionCor3
fi
log "MotionCor3 OK: $(command -v MotionCor3)"

# --- 4. CTFFIND 4.1.14 (CTF estimation) --------------------------------------
if ! command -v ctffind >/dev/null 2>&1; then
    log "installing CTFFIND 4.1.14"
    wget -q "$CTFFIND_URL" -O /tmp/ctffind.tar.gz
    mkdir -p "$SRC/ctffind" && tar -xzf /tmp/ctffind.tar.gz -C "$SRC/ctffind"
    sudo install -m 0755 "$(find "$SRC/ctffind" -type f -name ctffind | head -1)" /usr/local/bin/ctffind
fi
log "CTFFIND OK: $(command -v ctffind)"

# --- 5. RELION 5.0 (optional; CUDA, no GUI, no Python modules) ----------------
if [ "$INSTALL_RELION" = "true" ] && [ ! -x /opt/relion/bin/relion_refine ]; then
    log "building RELION 5.0 (10-30 min)"
    [ -d "$SRC/relion" ] || git clone -q -b ver5.0 https://github.com/3dem/relion.git "$SRC/relion"
    mkdir -p "$SRC/relion/build"
    ( cd "$SRC/relion/build" && \
      cmake .. -DCUDA=ON -DGUI=OFF -DFETCH_WEIGHTS=OFF -DCMAKE_BUILD_TYPE=Release \
               -DCMAKE_INSTALL_PREFIX=/opt/relion -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
               >"$SRC/relion-cmake.log" 2>&1 && \
      make -j"$(nproc)" >"$SRC/relion-build.log" 2>&1 && \
      sudo make install >>"$SRC/relion-build.log" 2>&1 ) \
        || { log "WARNING: RELION build failed (see $SRC/relion-build.log); MotionCor3/CTFFIND still usable"; \
             tail -20 "$SRC/relion-build.log" 2>/dev/null || tail -20 "$SRC/relion-cmake.log"; }
fi
[ -x /opt/relion/bin/relion_refine ] && log "RELION OK: /opt/relion/bin"

# --- 6. Python env for the before/after previews ------------------------------
if [ ! -x "$HOME/cryoem-venv/bin/python" ]; then
    log "python venv ~/cryoem-venv"
    python3 -m venv "$HOME/cryoem-venv"
fi
"$HOME/cryoem-venv/bin/pip" install -q --upgrade pip
"$HOME/cryoem-venv/bin/pip" install -q numpy mrcfile tifffile matplotlib

# --- 7. CephFS project volume (input data, used in place under /mnt) ----------
find_vol() {
    for d in /mnt/cephfs/*/*/*_cryoem-without-borders; do
        mountpoint -q "$d" 2>/dev/null && { echo "$d"; return; }
    done
}
VOL=$(find_vol || true)
if [ -z "$VOL" ]; then
    log "storage volume not mounted — re-running the CephFS mount script(s)"
    for s in "$HOME"/.ceph/*/mount_*.sh; do [ -f "$s" ] && sudo bash "$s" >/dev/null 2>&1 || true; done
    VOL=$(find_vol || true)
fi
if [ -z "$VOL" ]; then
    log "ERROR: cryoem-without-borders CephFS volume is not mounted under /mnt/cephfs"
    exit 4
fi
log "storage volume mounted at $VOL"

log "DONE"
