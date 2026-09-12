# =============================================================================
#  Trance AutoDJ & Visual Broadcaster
#
#  One self-contained image. It carries ffmpeg, Liquidsoap and Icecast so that
#  "docker run" on any machine with Docker Desktop is genuinely all it takes —
#  no host packages, no Ubuntu required on the host.
#
#  Base is ubuntu:26.04 deliberately: its archive has ffmpeg 8, Liquidsoap 2.4
#  and Python 3.14 together, and it is the same distribution as the reference
#  VM, so what passes here passes there.
#
#  Build:
#      docker build -f deploy/Dockerfile -t trance-autodj:latest .
#
#  Build with GPU support (Phase 5; adds several GB of CUDA wheels):
#      docker build -f deploy/Dockerfile --build-arg WITH_CUDA=1 \
#                   -t trance-autodj:cuda .
# =============================================================================

FROM ubuntu:26.04 AS base

# Ties the package on ghcr.io to this repository, so the package page shows
# the README and the repository lists the package. Without the source label
# they are two unrelated things that happen to share a name.
LABEL org.opencontainers.image.source="https://github.com/themechanic-dev/trance-autodj" \
      org.opencontainers.image.description="Self-hosted 24/7 trance station for YouTube Live: gapless playback, generative visuals, composes its own music." \
      org.opencontainers.image.licenses="MIT"

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# icecast2 asks debconf questions during install; answer them up front so the
# build never blocks. We generate our own icecast.xml at runtime anyway.
RUN echo 'icecast2 icecast2/icecast-setup boolean false' | debconf-set-selections \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.14 \
      python3.14-venv \
      ffmpeg \
      liquidsoap \
      icecast2 \
      ca-certificates \
      curl \
      tzdata \
      util-linux \
      procps \
 && rm -rf /var/lib/apt/lists/*

# Compile Liquidsoap's standard library into its system cache now, as root.
# Without this every start as an unprivileged user prints
# "Error while loading cache: Permission denied" and recompiles the stdlib —
# harmless, but it looks like a fault and it costs a second every restart.
RUN liquidsoap --cache-stdlib 2>/dev/null || \
    echo "stdlib cache unavailable; Liquidsoap will compile on each start"

RUN python3.14 -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Python dependencies, in their own layer so application edits do not
# re-download the world.
# -----------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt

ARG WITH_NUMBA=1
RUN if [ "$WITH_NUMBA" = "1" ]; then \
        pip install "numba>=0.61" || \
        echo "numba is unavailable for this Python; the generators will use numpy"; \
    fi

ARG WITH_CUDA=0
ARG CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN if [ "$WITH_CUDA" = "1" ]; then \
        pip install --index-url "$CUDA_INDEX_URL" torch \
     && pip install diffusers transformers accelerate safetensors; \
    fi

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
COPY app ./app
COPY scripts ./scripts
COPY config ./config
COPY pyproject.toml ./

# Runs unprivileged. The data volume must be writable by this uid; the
# entrypoint checks and says so plainly rather than failing obscurely.
# uid/gid 1000 on purpose: it is what a desktop Linux user has, so a bind
# mount of a host directory is writable without chown gymnastics. Ubuntu 26.04
# already ships a default 'ubuntu' account there, so reclaim the id first.
RUN existing_user="$(getent passwd 1000 | cut -d: -f1)" \
 && if [ -n "$existing_user" ]; then userdel --remove "$existing_user" 2>/dev/null || userdel "$existing_user"; fi \
 && existing_group="$(getent group 1000 | cut -d: -f1)" \
 && if [ -n "$existing_group" ]; then groupdel "$existing_group" 2>/dev/null || true; fi \
 && groupadd --gid 1000 autodj \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash autodj \
 && mkdir -p /data \
 && chown -R autodj:autodj /data /app

COPY --chown=autodj:autodj deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /app/scripts/*.sh /app/scripts/*.py

ENV TAD_APP__DATA_DIR=/data \
    TAD_APP__CONFIG_DIR=/app/config \
    TAD_APP__HOST=0.0.0.0 \
    TAD_APP__PORT=8080

VOLUME ["/data"]
EXPOSE 8080

USER autodj

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD /app/scripts/healthcheck.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
