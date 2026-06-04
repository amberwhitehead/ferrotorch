# Self-hosted GH Actions runner image for ferrotorch CUDA CI.
#
# Bakes a stable Rust toolchain + CUDA 13.0.2 dev tooling + the
# core-math-sys C build deps into one image so per-job cold starts
# skip the ~10-minute toolchain install.
#
# Built and used on a maintainer-controlled box (initial buildout:
# lucida). See `OPERATIONS.md` for the full deployment runbook.
#
# The image is intentionally minimal — no caching layer, no
# pre-fetched cargo registry. `Swatinem/rust-cache@v2` in the
# workflows handles per-job cargo cache.
#
# CUDA version pin: `CUDA_VERSION_TAG` argument lets the image stay
# in sync with `CUDARC_CUDA_VERSION` in the workflow env (currently
# 13020 = 13.0.2). Bump together when upgrading CUDA.

ARG CUDA_VERSION_TAG=13.0.2-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_VERSION_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    RUNNER_HOME=/home/runner \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    # ferray-ufunc -> core-math-sys vendored compoundf.c calls
    # exit(1) without #include <stdlib.h>; Clang 14+ promotes that
    # to an error per ISO C99. Match the workaround in macos-ci.yml
    # and linux-ci.yml.
    CFLAGS=-Wno-implicit-function-declaration

# Build deps: core-math-sys (via ferray-ufunc) needs gmp/mpfr.
# ca-certificates + curl for the runner tarball + rustup.
# jq for the registration-token API response.
# git for actions/checkout.
# build-essential + pkg-config + libssl-dev for native crates.
#
# .NET 6 runtime libs the actions/runner needs at startup. The
# bundled `bin/installdependencies.sh` script has hard-coded
# Ubuntu version cases up through 22.04 and falls through silently
# on 24.04, so we install the equivalent packages explicitly.
# libicu74 / liblttng-ust1t64 are the Ubuntu-24-renamed equivalents
# of Ubuntu-22's libicu70 / liblttng-ust1.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl jq git build-essential pkg-config \
      libgmp-dev libmpfr-dev libssl-dev \
      libclang-dev \
      sudo \
      libicu74 libkrb5-3 libssl3 zlib1g liblttng-ust1t64 \
    && rm -rf /var/lib/apt/lists/*

# Stable Rust (matches `dtolnay/rust-toolchain@stable` in the
# workflows). Includes clippy + rustfmt because the CUDA clippy
# step uses them.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- --default-toolchain stable --profile minimal -y \
      --component clippy --component rustfmt

# Non-root runner user. The runner's primary UID/GID is not significant
# (the container is ephemeral and the host fs is not mounted in besides
# the PAT secret), but the `gh-runner-secrets` GROUP must use a fixed
# GID that matches the host-side group owning /etc/ferrotorch-runner/pat
# (see OPERATIONS.md step 2). The bind-mount preserves the host file's
# GID; once GID 1500 is mapped to the `gh-runner-secrets` group inside
# the container, and the runner user is in that group, the read of
# the 0440 PAT resolves cleanly without making the file world-readable.
RUN groupadd --gid 1500 gh-runner-secrets \
    && useradd -m -d $RUNNER_HOME -s /bin/bash --groups gh-runner-secrets runner \
    && mkdir -p $RUNNER_HOME/_work \
    && chown -R runner:runner $RUNNER_HOME \
    && chown -R runner:runner /usr/local/rustup /usr/local/cargo

# GH Actions runner binary. Pinning a specific version because
# `latest` can shift mid-runbook; update via a deliberate image
# rebuild rather than implicit drift.
#
# Arch is detected via `uname -m` so the same Dockerfile builds
# cleanly on x86_64 (lucida-class hosts) and aarch64 (GB10-class
# hosts) — the `nvidia/cuda:*-ubuntu24.04` base is multi-arch so
# Docker picks the matching variant automatically. The runner
# tarball is NOT multi-arch, so we have to pick `x64` vs `arm64`
# explicitly.
ARG RUNNER_VERSION=2.319.1
RUN cd $RUNNER_HOME && \
    case "$(uname -m)" in \
        x86_64)  RUNNER_ARCH=x64 ;; \
        aarch64) RUNNER_ARCH=arm64 ;; \
        *) echo "unsupported architecture: $(uname -m)" && exit 1 ;; \
    esac && \
    curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz" && \
    tar xzf runner.tar.gz && \
    rm runner.tar.gz && \
    chown -R runner:runner $RUNNER_HOME

# Pre-install runner native deps (per `./bin/installdependencies.sh`)
# while still root.
RUN $RUNNER_HOME/bin/installdependencies.sh

USER runner
WORKDIR $RUNNER_HOME

COPY --chown=runner:runner entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
