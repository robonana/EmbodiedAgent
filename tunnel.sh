#!/bin/bash
# Robust auto-reconnecting SSH tunnel to the remote vLLM server.
#
# Forwards local :23333 -> remote localhost:23333 and automatically restarts
# whenever the connection drops. Keepalives detect a dead link in ~45s and
# force ssh to exit so the loop can reconnect (a plain `ssh -N -L` just hangs
# on a half-open TCP connection, which is why the tunnel felt "unstable").
#
# Usage:  bash tunnel.sh        # run in a terminal (or a tmux window) and leave it
#         Ctrl-C to stop.
set -u

REMOTE="chen@223.167.85.129"
SSH_PORT=50001
LOCAL_PORT=23333
REMOTE_PORT=23333

echo "[tunnel] local :$LOCAL_PORT  ->  $REMOTE (remote localhost:$REMOTE_PORT)"
echo "[tunnel] Ctrl-C to stop."

# Stop the reconnect loop cleanly on Ctrl-C.
trap 'echo; echo "[tunnel] stopped."; exit 0' INT TERM

while true; do
    # Free the local port in case a previous (half-dead) forward still holds it,
    # otherwise ExitOnForwardFailure would make every retry fail to bind.
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${LOCAL_PORT}/tcp" >/dev/null 2>&1 && sleep 1
    fi

    ssh -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=accept-new \
        -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
        -p "$SSH_PORT" "$REMOTE"

    code=$?
    echo "[tunnel] ssh exited (code $code) at $(date '+%F %T'); reconnecting in 3s…"
    sleep 3
done
