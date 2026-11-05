#!/bin/bash
# setup-receiver.sh — downstream receiver (FIU now, CAPE later): transfer tools + inbox.
# Results land on the receiver's LOCAL disk (~/cryoem-inbox), not on the Ceph volume.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get update -q >/dev/null
sudo -E apt-get install -y -q rsync iperf3 >/dev/null
mkdir -p "$HOME/cryoem-inbox"
echo "### SETUP: receiver ready: $(hostname), inbox ~/cryoem-inbox, $(df -h ~ | tail -1 | awk '{print $4}') free"
