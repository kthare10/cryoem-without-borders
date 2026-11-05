#!/bin/bash
# CryoEM Without Borders — slice bring-up (LoomAI run script).
#
# LoomAI's Run button calls this with the slice name as $1 and exports the
# weave.json args as env vars. It provisions the GPU node, mounts the
# cryoem-without-borders storage volume, installs MotionCor3 / CTFFIND4 / RELION,
# adds the FIU receiver node (stand-in for CAPE) on a dedicated L2 transfer link,
# and EXITS once the node is ready — it does not keep polling the slice.
# Run the experiment from cryoem_experiment.ipynb.
#
# Manual use:
#   ./weave.sh cryoem-demo                    # provision (default)
#   ACTION=status ./weave.sh cryoem-demo      # one-shot health check
#   ACTION=extend ./weave.sh cryoem-demo      # renew lease (LEASE_DAYS)
#   ACTION=stop   ./weave.sh cryoem-demo      # delete the slice
set -uo pipefail

SLICE_NAME="${SLICE_NAME:-${1:-cryoem-demo}}"
SLICE_NAME=$(echo "$SLICE_NAME" | sed 's/[^a-zA-Z0-9-]/-/g; s/--*/-/g; s/^-//; s/-$//')
ACTION="${ACTION:-start}"

export SITE="${SITE:-}"
export GPU_MODEL="${GPU_MODEL:-auto}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CORES="${CORES:-16}"
export RAM="${RAM:-64}"
export DISK="${DISK:-500}"
export CUDA_VERSION="${CUDA_VERSION:-12.4}"
export INSTALL_RELION="${INSTALL_RELION:-true}"
export LEASE_DAYS="${LEASE_DAYS:-14}"
export KEEP_SLICE="${KEEP_SLICE:-true}"
export EXCLUDE_SITES="${EXCLUDE_SITES:-}"
export RECEIVER_SITE="${RECEIVER_SITE:-FIU}"
export RECEIVER_CORES="${RECEIVER_CORES:-8}"
export RECEIVER_RAM="${RECEIVER_RAM:-32}"
export RECEIVER_DISK="${RECEIVER_DISK:-500}"
export XFER_SUBNET="${XFER_SUBNET:-192.168.100.0/24}"
export XFER_NIC="${XFER_NIC:-NIC_Basic}"
export PIN_CPU="${PIN_CPU:-true}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="python3"
[ -x "$SCRIPT_DIR/.venv/bin/python" ] && PY="$SCRIPT_DIR/.venv/bin/python"

# Stopped mid-provision: delete the partial slice only when KEEP_SLICE=false.
on_stop() {
    if [ "$KEEP_SLICE" = "true" ]; then
        echo "### PROGRESS: Stop requested — KEEP_SLICE=true, leaving slice '$SLICE_NAME' in place."
    else
        echo "### PROGRESS: Stop requested — deleting slice '$SLICE_NAME'..."
        "$PY" "$SCRIPT_DIR/provision.py" stop "$SLICE_NAME" 2>&1 || true
    fi
    exit 0
}
trap on_stop SIGTERM SIGINT

if [ "$ACTION" = "start" ]; then
    echo "### PROGRESS: === CryoEM Without Borders — bring-up ==="
    echo "### PROGRESS: Slice: $SLICE_NAME | GPUs: ${NUM_GPUS}x $GPU_MODEL | Site: ${SITE:-auto}"
    echo "### PROGRESS: VM: ${CORES}c/${RAM}G/${DISK}G | CUDA $CUDA_VERSION | RELION: $INSTALL_RELION"
    echo "### PROGRESS: Storage: cryoem-without-borders (CephFS, input) | Keep slice: $KEEP_SLICE"
    echo "### PROGRESS: Receiver: ${RECEIVER_SITE:-none} (${RECEIVER_CORES}c/${RECEIVER_RAM}G/${RECEIVER_DISK}G) via L2 link $XFER_SUBNET | NIC: $XFER_NIC | CPU pinning: $PIN_CPU"
fi

cd "$SCRIPT_DIR"
"$PY" "$SCRIPT_DIR/provision.py" "$ACTION" "$SLICE_NAME" &
wait $!
RC=$?

if [ "$ACTION" = "start" ] && [ $RC -eq 0 ]; then
    echo "### PROGRESS: Next: open cryoem_experiment.ipynb and run it against '$SLICE_NAME'."
    echo "### PROGRESS: The slice stays up. Delete it with: ACTION=stop ./weave.sh $SLICE_NAME"
fi
exit $RC
