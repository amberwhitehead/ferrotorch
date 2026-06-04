# Self-hosted GPU runner — operations runbook

This document is the deployment runbook for the ferrotorch self-hosted
GitHub Actions runner that powers `cuda-ci.yml` and `nightly.yml`. It is
intentionally complete enough that a maintainer who has never touched this
before can stand up a runner end-to-end without external context.

Initial buildout target: `lucida` (a maintainer-controlled CUDA dev box).

Tracking: [issue #32](https://github.com/forecast-bio/ferrotorch/issues/32).

## What this gets you

- A Docker container running a single GitHub Actions runner with CUDA
  access, registered against `forecast-bio/ferrotorch` with labels
  `[self-hosted, linux, x64, gpu, cuda]`.
- Ephemeral lifecycle: one container = one job. The container exits
  after the job, systemd respawns it, and the new container re-registers
  fresh. Nothing from job A is visible to job B.
- Combined with the `gpu` GitHub Environment's required-reviewer gate,
  every workflow run on this runner requires explicit maintainer approval
  via the GitHub Actions UI before any code from the PR executes on the
  host.

## Threat model

The runner host runs arbitrary CI code from PR branches. Defenses, in
the order they fire:

1. **GitHub Environment gate** (`environment: gpu`): GitHub holds the job
   until a maintainer clicks "Approve and deploy". No code reaches the
   runner before approval.
2. **Container isolation**: the runner executes inside a Docker container.
   The host filesystem is not mounted in (except the read-only PAT secret
   at `/run/secrets/pat`).
3. **Ephemeral runner**: the container exits after one job and is recreated.
   No state — caches, build artifacts, runner registration — survives.
4. **Resource limits**: cgroup caps prevent a runaway job from DoSing the
   host (`--memory=24g`, `--cpus=8`, `--pids-limit=2048`).
5. **Privilege drop**: container runs with `--cap-drop=ALL` and
   `--security-opt=no-new-privileges`; the in-container user is unprivileged.
6. **No persistent paths**: the runner's writable area lives in tmpfs
   (`/home/runner/_work`, `/usr/local/cargo/registry`, `/tmp`); the
   container rootfs itself is writable (necessary for the runner's own
   `.env` / `.path` / `.credentials` state files), but the whole
   container is `--rm`'d on every exit, so nothing persists across
   jobs. `--read-only` was tried initially but conflicts with the
   actions/runner's state-file layout; ephemerality is the substitute.

Operational defenses outside this runbook (configure in the GitHub UI):

- `Settings → Actions → General → Fork pull request workflows`: require
  approval for outside collaborators (default; verify it's on).
- `Settings → Environments → gpu`: required reviewers + optional deployment-
  branch restriction so forks can't trigger paid runs.

## Quick-start (the impatient path)

If host prerequisites are already in place (NVIDIA driver, Docker,
nvidia-container-toolkit) and the PAT is in your hand, the
`just ci-runner` recipes wrap the deployment. From the repo root:

```bash
# One-time: set up host-side group + secret. The justfile checks
# both pre-flight, so run these before `just ci-runner up`.
sudo groupadd --gid 1500 gh-runner-secrets
sudo mkdir -p /etc/ferrotorch-runner
sudo chmod 0700 /etc/ferrotorch-runner
echo "<PAT>" | sudo tee /etc/ferrotorch-runner/pat > /dev/null
sudo chmod 0440 /etc/ferrotorch-runner/pat
sudo chown root:gh-runner-secrets /etc/ferrotorch-runner/pat

# Idempotent install + start.
just ci-runner up

# Day-to-day:
just ci-runner status
just ci-runner logs
just ci-runner restart
just ci-runner build  # after a Dockerfile change
just ci-runner down   # stop without uninstalling
```

The detailed walkthrough below is the source of truth — read it once to
understand what `just ci-runner up` actually does. The justfile is the
shortcut after that.

## Host prerequisites

Tested on Ubuntu 24.04 LTS. Adapt commands for your distribution.

### 1. NVIDIA driver

```bash
nvidia-smi
```

Must work on the host before anything else. If it doesn't, install the
appropriate driver package (e.g., `sudo apt install nvidia-driver-550`).

### 2. Docker (official packages)

The NVIDIA Container Toolkit's apt repo provides `containerd.io`, which
conflicts with Ubuntu's `docker.io` package. Use Docker's official packages
to avoid that:

```bash
# Docker's GPG key + repo
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
    https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
```

### 3. NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Verify GPU passthrough

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

This must list your GPU(s). If it doesn't, stop and fix the host setup
before continuing — none of the rest will work.

## Runner setup

### 1. Mint a registration PAT

GitHub UI: <https://github.com/settings/personal-access-tokens/new>.

- Type: **Fine-grained personal access token**.
- Resource owner: your account (or the org, if delegated).
- Repository access: **Only select repositories** → `forecast-bio/ferrotorch`.
- Repository permissions:
  - `Administration` → **Read and write** (just for the
    `actions/runners/registration-token` endpoint).
  - `Actions` → **Read**.
- Expiration: 90 days max. Set a calendar reminder for rotation.
- Generate, copy the value immediately (GitHub will not show it again).

Smoke-test the PAT before depositing it on the host:

```bash
curl -fsSL -X POST \
    -H "Authorization: Bearer <THE_PAT>" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/forecast-bio/ferrotorch/actions/runners/registration-token \
    | jq .
```

Expected response: a JSON object with `token` and `expires_at`. If you
get `401 Bad credentials` the PAT value is wrong; `403 Resource not
accessible` means it's missing the `Administration: read+write`
permission.

### 2. Store the PAT on the host

The runner container reads the PAT as a non-root user (image's `runner`
user). Rather than make the host file world-readable, we use a fixed-GID
group that's defined on both sides of the bind mount: the host file is
group-owned by `gh-runner-secrets` (mode 0440), the in-container
`runner` user is a member of the same-GID group (created in the
Dockerfile at GID 1500), and the kernel's bind-mount preserves the GID
so the group-read bit applies.

```bash
# 1. Create the host group with the fixed GID baked into the Dockerfile.
sudo groupadd --gid 1500 gh-runner-secrets

# 2. Create the secret directory.
sudo mkdir -p /etc/ferrotorch-runner
sudo chmod 0700 /etc/ferrotorch-runner
sudo chown root:root /etc/ferrotorch-runner

# 3. Deposit the PAT. Use `sudo tee` (not `echo > file`) so the PAT
#    doesn't leak into shell history; the redirect runs as root.
echo "<THE_PAT>" | sudo tee /etc/ferrotorch-runner/pat > /dev/null

# 4. Tighten perms — root owns + writes, group `gh-runner-secrets`
#    reads, everyone else gets nothing.
sudo chmod 0440 /etc/ferrotorch-runner/pat
sudo chown root:gh-runner-secrets /etc/ferrotorch-runner/pat
```

Verify:

```bash
sudo ls -la /etc/ferrotorch-runner/
# expected:
#   drwx------ 2 root root              ... .
#   -r--r----- 1 root gh-runner-secrets ... pat

# Your normal user should still hit Permission denied because the
# parent dir is 0700 root:root — no traversal:
cat /etc/ferrotorch-runner/pat
# expected: Permission denied
```

If GID 1500 is already in use on this host (`getent group 1500`
returns something else), pick another unused GID. You'll need to
match it in both places: the host `groupadd --gid <N>` above AND
the Dockerfile's `groupadd --gid <N> gh-runner-secrets` line.

### 3. Build the runner image

Clone the repo (or pull on an existing clone), then build the runner
image. The image is local-only — never push it to a registry, since
nothing in it is repo-specific and the source of truth is the Dockerfile.

```bash
git clone https://github.com/forecast-bio/ferrotorch.git
cd ferrotorch/ci/runner
docker build -t ferrotorch-runner:latest .
```

The image is ~6 GB (the CUDA dev image is most of it). Build takes
~10 min on a cold cache. Rebuilds (e.g., to pull a newer GH Actions
runner version, bump CUDA, etc.) are cheap thanks to Docker's layer
cache.

### 4. Install the systemd unit

```bash
sudo cp ci/runner/ferrotorch-runner.service \
    /etc/systemd/system/ferrotorch-runner.service
sudo systemctl daemon-reload
sudo systemctl enable ferrotorch-runner.service
```

Don't start it yet — we want to confirm the GitHub Environment is set
up first (see next section).

### 5. Configure the GitHub Environment

GitHub UI: `Settings → Environments → New environment → name = gpu`.

- **Required reviewers**: add maintainers authorised to approve GPU runs.
  Free plans allow up to 6 reviewers per environment; one approval is
  enough to release a run.
- **Wait timer**: 0 (or 2-5 min if you want a cancellation window).
- **Deployment branches and tags**: "Selected branches and tags".
  Add `main`, `release/*`, and `refs/pull/*/merge` (the last is what lets
  PR runs target this environment).
- **Environment secrets**: none needed initially.

### 6. Start the runner

```bash
sudo systemctl start ferrotorch-runner.service
sudo systemctl status ferrotorch-runner.service
```

Within ~30 seconds the runner should register itself. Check the GitHub
UI at `Settings → Actions → Runners` — you should see one runner with
labels `[self-hosted, linux, x64, gpu, cuda]` and a status of "Idle".

### 7. Test fire

Open a trivial PR (whitespace change). The `CUDA CI` check appears with
status "Waiting for review". Approve it via the Actions UI. The runner
spins up the job. Confirm:

- The job appears assigned to your runner in the Actions UI.
- `docker ps` on the host shows a `ferrotorch-runner-*` container.
- `nvidia-smi` on the host shows GPU activity.
- After the job finishes, the container is gone (`docker ps` empty),
  systemd has respawned it, and a fresh runner is re-registered in the
  GitHub Runners list (with a different name).

If anything goes wrong, see `journalctl -u ferrotorch-runner.service`
for the host-side log, and the Actions UI for the runner-side log.

## Ongoing operations

### Monitoring

```bash
# Service state on the host
sudo systemctl status ferrotorch-runner.service

# Recent logs from the systemd unit (last 100 lines)
sudo journalctl -u ferrotorch-runner.service -n 100 --no-pager

# Live tail
sudo journalctl -u ferrotorch-runner.service -f
```

The GH Actions UI is the source of truth for run history; the systemd
logs are useful when the runner itself misbehaves (registration
failures, Docker errors, GPU issues).

### Rotating the PAT

Every 90 days (calendar-reminder):

1. Generate a new fine-grained PAT (same scope as the original).
2. Overwrite the file:
   ```bash
   echo "<NEW_PAT>" | sudo tee /etc/ferrotorch-runner/pat > /dev/null
   ```
3. Restart the runner:
   ```bash
   sudo systemctl restart ferrotorch-runner.service
   ```
4. Revoke the old PAT in the GitHub UI.

No Docker rebuild needed; the entrypoint reads the PAT fresh each
container start.

### Updating the runner version

To pick up a newer `actions/runner` release:

1. Edit `ci/runner/Dockerfile` and bump `ARG RUNNER_VERSION`.
2. Rebuild: `docker build -t ferrotorch-runner:latest ci/runner/`.
3. Restart: `sudo systemctl restart ferrotorch-runner.service`.

The currently-running job (if any) is not interrupted — the new image is
picked up on the next container spawn after the in-flight job completes.

### Upgrading CUDA

1. Pick a new tag from <https://hub.docker.com/r/nvidia/cuda>.
2. Edit `ci/runner/Dockerfile`: bump `ARG CUDA_VERSION_TAG`.
3. Update the `CUDARC_CUDA_VERSION` env in `cuda-ci.yml` and
   `nightly.yml` to match (e.g., `13020` for 13.0.2; `13030` for
   13.0.3).
4. Rebuild + restart as above.

### Decommissioning

1. Stop and disable the service:
   ```bash
   sudo systemctl disable --now ferrotorch-runner.service
   ```
2. The ephemeral runner already deregisters itself on each container
   exit. If the current registration is stale, remove it manually via
   the GitHub UI (`Settings → Actions → Runners → ferrotorch-runner-*
   → Remove`) or via API:
   ```bash
   REMOVE_TOKEN=$(curl -fsSL -X POST \
       -H "Authorization: Bearer $(sudo cat /etc/ferrotorch-runner/pat)" \
       -H "Accept: application/vnd.github+json" \
       https://api.github.com/repos/forecast-bio/ferrotorch/actions/runners/remove-token \
       | jq -r .token)
   # Then run config.sh remove --token "$REMOVE_TOKEN" from inside the
   # runner directory if needed.
   ```
3. Revoke the PAT in the GitHub UI.
4. Remove host state:
   ```bash
   sudo rm -rf /etc/ferrotorch-runner /etc/systemd/system/ferrotorch-runner.service
   sudo systemctl daemon-reload
   docker rmi ferrotorch-runner:latest
   ```

## Troubleshooting

**Symptom: runner registers but jobs never get assigned to it.**

The `gpu` environment's deployment-branch list probably excludes
`refs/pull/*/merge`. Add it via `Settings → Environments → gpu →
Deployment branches`.

**Symptom: `FATAL: failed to mint registration token from PAT`** in the
journal log.

PAT lacks `Administration: read+write` repo permission. Edit the PAT
in the GitHub UI (no need to regenerate; permissions are editable in
place), then restart the systemd unit.

**Symptom: jobs fail with `nvidia-smi: command not found` inside the
container.**

NVIDIA Container Toolkit isn't configured. Re-run
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl
restart docker`, and verify with the `docker run --gpus all` smoke test.

**Symptom: out-of-disk on the host.**

Each ephemeral runner reuses the same Docker image but its tmpfs
mounts (`/home/runner/_work`, `/tmp`, `/usr/local/cargo/registry`)
live in host RAM. If you don't have ~32 GB of RAM available for tmpfs,
adjust the `size=` values on those tmpfs lines in the systemd unit,
or replace tmpfs with named Docker volumes (less safe — volumes
persist across container restarts, breaking ephemerality).

**Symptom: workflow doesn't show a `CUDA CI` check at all.**

Linux-CI's `linux-ci.yml` runs on every PR (no environment); the
`cuda-ci.yml` workflow only triggers on `pull_request` against `main`.
If the PR's base branch is something else, it won't fire. Use
`workflow_dispatch` from the Actions tab for one-off runs.
