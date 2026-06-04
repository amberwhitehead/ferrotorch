#!/usr/bin/env bash
# Ephemeral GH Actions runner registration + run loop for one job.
#
# Lifecycle (one container = one job):
#   1. Read the registration PAT from the read-only secret mount.
#   2. POST to the repo's actions/runners/registration-token endpoint
#      to mint a one-shot 1-hour-lived registration token.
#   3. Register this runner as ephemeral with labels [self-hosted,
#      linux, x64, gpu, cuda].
#   4. Run one job. The `--ephemeral` flag means `./run.sh` exits
#      cleanly after one job completes, regardless of success.
#   5. (best-effort) Remove the runner registration to keep the
#      GitHub-side runner list clean.
#
# systemd's `Restart=always` policy then spawns the next container,
# which repeats the loop. This is the canonical "one-shot ephemeral
# self-hosted runner" pattern recommended in the GH docs.

set -euo pipefail

REPO_URL="https://github.com/forecast-bio/ferrotorch"
PAT_FILE="${PAT_FILE:-/run/secrets/pat}"
RUNNER_LABELS="self-hosted,linux,x64,gpu,cuda"

# Validate the PAT mount before doing anything that needs the network.
if [[ ! -r "$PAT_FILE" ]]; then
    echo "FATAL: PAT secret not readable at $PAT_FILE" >&2
    echo "       Check the systemd unit's -v mount + the host file's perms (0400)." >&2
    exit 1
fi

PAT=$(< "$PAT_FILE")
# Trim trailing newline that may have crept in from `echo "..." | tee`
PAT="${PAT%$'\n'}"

REPO_API="https://api.github.com/repos/forecast-bio/ferrotorch"

mint_token() {
    local endpoint="$1"  # registration-token or remove-token
    curl -fsSL -X POST \
        -H "Authorization: Bearer ${PAT}" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "${REPO_API}/actions/runners/${endpoint}" \
        | jq -r .token
}

cleanup() {
    # Best-effort: try to mint a remove-token and deregister. If the
    # runner already removed itself (which it does on a clean ephemeral
    # exit), this is a no-op. Errors are swallowed — we don't want a
    # cleanup hiccup to crash-loop the systemd unit.
    if [[ -f "${RUNNER_HOME}/.runner" ]]; then
        local remove_token
        remove_token=$(mint_token remove-token || true)
        if [[ -n "$remove_token" ]]; then
            ./config.sh remove --token "$remove_token" || true
        fi
    fi
}
trap cleanup EXIT

# Mint registration token + register the runner as ephemeral.
REG_TOKEN=$(mint_token registration-token)
if [[ -z "$REG_TOKEN" || "$REG_TOKEN" == "null" ]]; then
    echo "FATAL: failed to mint registration token from PAT." >&2
    echo "       Check the PAT has 'Administration: read+write' on $REPO_URL." >&2
    exit 1
fi

# Name the runner with the container hostname + a short timestamp so
# each ephemeral incarnation is distinguishable in the GH Runners UI.
RUNNER_NAME="${HOSTNAME}-$(date +%s)"

./config.sh \
    --url "$REPO_URL" \
    --token "$REG_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --work _work \
    --ephemeral \
    --unattended \
    --replace

# Run exactly one job, then exit. --ephemeral makes this terminate
# after the first job (regardless of success); systemd respawns.
exec ./run.sh
