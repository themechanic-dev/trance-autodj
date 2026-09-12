#!/bin/bash
# Container entry point.
#
#   serve      (default) run the web application
#   preflight  run the environment check and exit
#   generator  run the visual generator worker
#   streamer   explains why there is no separate streamer process
#   shell      drop into bash
#
# Any other argument is executed verbatim, so "docker run ... python -c ..."
# keeps working.
set -euo pipefail

DATA_DIR="${TAD_APP__DATA_DIR:-/data}"

# A fresh volume or a path that does not exist yet is normal, not an error;
# only report a problem if we cannot make it and cannot write to it.
mkdir -p "$DATA_DIR" 2>/dev/null || true

if [ ! -w "$DATA_DIR" ]; then
  cat >&2 <<EOF
The data directory $DATA_DIR is not writable by uid $(id -u).

If you mounted a host directory, give it to this container's user:
    sudo chown -R 1000:1000 /path/on/host
or run the container with:
    --user "\$(id -u):\$(id -g)"
EOF
  exit 1
fi

command="${1:-serve}"
shift || true

# The healthcheck runs as a separate `docker exec` and inherits the image's
# environment, not anything this script exports, so the role has to be left
# somewhere it can read. A tmpfs file is enough and needs no privileges.
echo "$command" > /tmp/tad-role 2>/dev/null || true

case "$command" in
  serve)
    # app.main:main() starts uvicorn with the host/port from the resolved
    # configuration, so TAD_* overrides and config.yaml agree with each other
    # instead of the entrypoint having a second opinion.
    exec python -m app.main "$@"
    ;;
  preflight)
    exec python scripts/preflight.py "$@"
    ;;
  generator)
    export TAD_SERVICE=generator
    exec python -m app.services.visual.worker "$@"
    ;;
  streamer)
    cat >&2 <<'EOF'
There is no separate streamer process. The web application owns ffmpeg, which
is what lets the dashboard start and stop the broadcast and lets the watchdog
restart it after a failure.

Run "serve" and go live from the Stream page, or POST /api/stream/start.
EOF
    exit 64
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac
