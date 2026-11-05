#!/bin/bash
# net-tune.sh IP [MTU] — WAN TCP tuning + MTU on the interface that holds IP (the XFER link).
set -euo pipefail
IP="$1"; MTU="${2:-9000}"
sudo modprobe tcp_bbr 2>/dev/null || true
sudo sysctl -q -w net.core.rmem_max=268435456 net.core.wmem_max=268435456 \
    net.ipv4.tcp_rmem="4096 87380 134217728" net.ipv4.tcp_wmem="4096 65536 134217728" \
    net.core.default_qdisc=fq net.ipv4.tcp_congestion_control=bbr net.ipv4.tcp_mtu_probing=1
DEV=$(ip -o -4 addr show | awk -v ip="$IP" 'index($4, ip"/")==1 {print $2}')
[ -n "$DEV" ] || { echo "no interface holds $IP" >&2; exit 1; }
sudo ip link set dev "$DEV" mtu "$MTU"
echo "DEV=$DEV MTU=$MTU"
