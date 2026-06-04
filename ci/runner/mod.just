# ferrotorch self-hosted GH Actions runner — operations recipes.
#
# Invoke from the repo root via `just ci-runner <recipe>`:
#
#   just ci-runner up          # idempotent install + start
#   just ci-runner down        # stop + disable (keeps image + unit + PAT)
#   just ci-runner restart
#   just ci-runner status
#   just ci-runner logs        # last 50 journal lines
#   just ci-runner follow      # tail -f the journal
#   just ci-runner build       # rebuild docker image (after Dockerfile change)
#   just ci-runner test        # interactive dry-run of the container
#   just ci-runner purge       # full uninstall (DESTRUCTIVE — needs --force)
#
# All recipes shell out to `sudo` where needed and will prompt for
# your sudo password. Designed for Ubuntu 24.04 hosts that have
# completed the prerequisite steps in OPERATIONS.md (NVIDIA driver,
# Docker, nvidia-container-toolkit installed; PAT generated and
# deposited at /etc/ferrotorch-runner/pat).
#
# The recipes are intentionally re-runnable: an `up` after an
# already-running runner reconciles state (rebuild + reinstall +
# restart). No errors on idempotent no-ops.

set shell := ["bash", "-euo", "pipefail", "-c"]

# Path resolution note: when just invokes a recipe from a module
# (`just ci-runner <recipe>`), it sets the recipe's working directory
# to the MODULE FILE's directory — i.e. `ci/runner/`. Relative paths
# below are therefore module-relative (the Dockerfile and unit file
# sit alongside this mod.just), not repo-root-relative. Don't reach
# for `justfile_directory()` (that returns the *root* justfile's dir
# = repo root) or `ci/runner/...` (would double-prefix to
# `ci/runner/ci/runner/...`).
IMAGE        := "ferrotorch-runner:latest"
SERVICE      := "ferrotorch-runner.service"
SECRET_DIR   := "/etc/ferrotorch-runner"
SECRET_FILE  := "/etc/ferrotorch-runner/pat"
SECRET_GROUP := "gh-runner-secrets"
SECRET_GID   := "1500"
UNIT_SRC     := "ferrotorch-runner.service"
UNIT_DST     := "/etc/systemd/system/ferrotorch-runner.service"
DOCKER_CTX   := "."

# Default recipe — list the module's recipes.
default:
    @just --list --justfile mod.just

# Idempotent install: validate prereqs, build image, install unit,
# enable, start. Safe to re-run.
up: _check-prereqs _check-secret build _install-unit
    @echo ">>> enabling + starting {{SERVICE}}"
    sudo systemctl daemon-reload
    sudo systemctl enable {{SERVICE}}
    sudo systemctl restart {{SERVICE}}
    @echo
    @echo ">>> waiting 5s for the runner to register with GitHub..."
    @sleep 5
    @just ci-runner status

# Stop + disable the runner but leave artifacts (image, unit file,
# PAT secret, host group) in place. Use this for short-term
# maintenance; use `purge` for a full uninstall.
down:
    @echo ">>> stopping + disabling {{SERVICE}}"
    -sudo systemctl stop {{SERVICE}}
    -sudo systemctl disable {{SERVICE}}
    @# Best-effort: remove the still-running container if Restart=
    @# triggers a respawn between stop and disable.
    -sudo docker rm -f ferrotorch-runner 2>/dev/null
    @echo ">>> stopped. Artifacts retained (image, unit, secret). Re-run \`just ci-runner up\` to resume."

restart:
    sudo systemctl restart {{SERVICE}}
    @just ci-runner status

status:
    @echo ">>> systemctl status"
    -sudo systemctl status {{SERVICE}} --no-pager --lines 0
    @echo
    @echo ">>> container state"
    -sudo docker ps --filter name=ferrotorch-runner
    @echo
    @echo ">>> recent journal (last 10 lines)"
    -sudo journalctl -u {{SERVICE}} -n 10 --no-pager --output=cat

# Tail the last N journal lines.
logs N="50":
    sudo journalctl -u {{SERVICE}} -n {{N}} --no-pager

# Follow the journal live. Ctrl-C to exit.
follow:
    sudo journalctl -u {{SERVICE}} -f

# Rebuild the docker image. Picks up Dockerfile changes; running
# jobs are NOT interrupted (the new image is used on the next
# container spawn after the current job finishes).
build:
    @echo ">>> building {{IMAGE}}"
    sudo docker build -t {{IMAGE}} {{DOCKER_CTX}}

# Interactive dry-run of the runner container, attached to the
# terminal. Useful for debugging registration / startup. Ctrl-C
# to stop. Runs once, then exits (`--rm` cleans up).
test: _check-secret
    @echo ">>> dry-run: starting container interactively. Ctrl-C to stop."
    sudo docker run --rm \
        --name ferrotorch-runner \
        --hostname ferrotorch-runner \
        --gpus all \
        --memory=24g --cpus=8 --pids-limit=2048 \
        --tmpfs /tmp:rw,size=2g \
        --tmpfs /home/runner/_work:rw,size=20g,exec \
        --tmpfs /home/runner/_diag:rw,size=128m \
        --tmpfs /usr/local/cargo/registry:rw,size=10g \
        --cap-drop=ALL --security-opt=no-new-privileges \
        --env PAT_FILE=/run/secrets/pat \
        -v {{SECRET_FILE}}:/run/secrets/pat:ro \
        {{IMAGE}}

# DESTRUCTIVE — full uninstall. Stops + disables the service,
# removes the systemd unit, the docker image, the PAT secret + its
# parent dir, and the host group. Requires `--force` so accidental
# `just ci-runner purge` doesn't nuke a working setup.
purge force="":
    @if [ "{{force}}" != "--force" ]; then \
        echo "purge is destructive (removes systemd unit, docker image,"; \
        echo "PAT secret + dir, gh-runner-secrets group). Re-run as:"; \
        echo "  just ci-runner purge --force"; \
        exit 1; \
    fi
    @echo ">>> purging runner artifacts"
    -sudo systemctl stop {{SERVICE}}
    -sudo systemctl disable {{SERVICE}}
    -sudo docker rm -f ferrotorch-runner
    -sudo rm -f {{UNIT_DST}}
    sudo systemctl daemon-reload
    -sudo docker rmi {{IMAGE}}
    -sudo rm -rf {{SECRET_DIR}}
    -sudo groupdel {{SECRET_GROUP}}
    @echo ">>> purge complete. The PAT itself should also be revoked"
    @echo "    in the GitHub UI (Settings -> Developer settings ->"
    @echo "    Personal access tokens)."

# ── internal helpers (prefixed `_`, not for direct invocation) ───────────

_check-prereqs:
    @command -v docker >/dev/null || { echo "FATAL: docker not installed. See OPERATIONS.md prerequisites."; exit 1; }
    @command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not on PATH. Install the NVIDIA driver."; exit 1; }
    @sudo docker info 2>/dev/null | grep -q "Runtimes:.*nvidia" \
        || { echo "FATAL: docker isn't configured for the nvidia runtime. Run \`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker\`."; exit 1; }
    @getent group {{SECRET_GROUP}} | grep -q "^{{SECRET_GROUP}}:x:{{SECRET_GID}}:" \
        || { echo "FATAL: host group {{SECRET_GROUP}} (GID {{SECRET_GID}}) not present. Run \`sudo groupadd --gid {{SECRET_GID}} {{SECRET_GROUP}}\`."; exit 1; }

_check-secret:
    @sudo test -f {{SECRET_FILE}} \
        || { echo "FATAL: PAT not found at {{SECRET_FILE}}. See OPERATIONS.md step 2."; exit 1; }
    @perms=$(sudo stat -c '%a %U %G' {{SECRET_FILE}}); \
     if [ "$perms" != "440 root {{SECRET_GROUP}}" ]; then \
         echo "FATAL: PAT has wrong perms/owner. Got: $perms; want: 440 root {{SECRET_GROUP}}"; \
         echo "Fix: sudo chmod 0440 {{SECRET_FILE}} && sudo chown root:{{SECRET_GROUP}} {{SECRET_FILE}}"; \
         exit 1; \
     fi

_install-unit:
    @echo ">>> installing {{SERVICE}}"
    sudo install -m 0644 -o root -g root {{UNIT_SRC}} {{UNIT_DST}}
