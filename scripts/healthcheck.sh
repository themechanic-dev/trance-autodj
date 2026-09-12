#!/bin/sh
# Container/systemd healthcheck.
#
# The image runs more than one kind of process and they cannot be checked the
# same way. The web application answers /health, which does no work, so a busy
# generator cannot make it look unhealthy. The generator has no socket at all:
# for it, healthy means the worker is still running. Without this split, a
# generator container started from the same image — which is exactly what
# `docker compose run autodj generator` does — is reported unhealthy for its
# entire life, and an unhealthy container is not just cosmetic: it is what
# restart policies and monitoring act on.
set -eu

# Written by the entrypoint; the environment we would rather read is not
# visible from here.
role=$(cat /tmp/tad-role 2>/dev/null || echo serve)

case "$role" in
  generator)
    # pgrep never matches itself, so this is the worker or nothing.
    pgrep -f 'app\.services\.visual\.worker' > /dev/null
    ;;
  *)
    PORT="${TAD_APP__PORT:-8080}"
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health" > /dev/null
    ;;
esac
