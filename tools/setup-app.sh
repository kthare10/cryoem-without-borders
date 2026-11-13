#!/bin/bash
# setup-app.sh - install the CryoEM demo web app on the GPU node as a systemd service.
#
#   bash ~/tools/setup-app.sh [PORT]        (as the login user; uses sudo for systemd)
#
# Serves 127.0.0.1:PORT (default 8080) only: open it with a LoomAI tunnel to node
# "cryoem", port PORT, or an SSH tunnel. Runs as the login user (it reads ~/cryoem-work,
# starts runs, and uses the 'receiver' ssh alias). Uses ~/cryoem-venv (numpy, mrcfile,
# matplotlib from setup-node.sh) plus flask and waitress. Idempotent.
set -euo pipefail
PORT="${1:-8080}"
SRC="$(cd "$(dirname "$0")" && pwd)/demo_app"
DEST="$HOME/cryoem-app"
PY="$HOME/cryoem-venv/bin/python"

[ -x "$PY" ] || { echo "ERROR: $PY not found - run setup-node.sh first" >&2; exit 2; }
mkdir -p "$DEST/static" "$HOME/cryoem-work/app"
install -m 0644 "$SRC/app.py" "$SRC/index.html" "$DEST/"
install -m 0644 "$SRC"/static/* "$DEST/static/"
"$PY" -m pip install -q 'flask>=2.2,<4' 'waitress>=2.1,<4'

sudo tee /etc/systemd/system/cryoem-demo-app.service >/dev/null <<UNIT
[Unit]
Description=CryoEM Without Borders demo app (127.0.0.1:$PORT)
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$DEST
Environment=HOME=$HOME
ExecStart=$PY $DEST/app.py --host 127.0.0.1 --port $PORT
Restart=on-failure
# Runs started from the app must outlive an app restart.
KillMode=process

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable cryoem-demo-app.service >/dev/null 2>&1
sudo systemctl restart cryoem-demo-app.service
for _ in $(seq 1 15); do
  curl -fsS "http://127.0.0.1:$PORT/api/info" >/dev/null 2>&1 && { echo "== demo app up on 127.0.0.1:$PORT"; exit 0; }
  sleep 1
done
echo "ERROR: demo app did not answer on 127.0.0.1:$PORT" >&2
sudo journalctl -u cryoem-demo-app.service -n 30 --no-pager >&2 || true
exit 1
