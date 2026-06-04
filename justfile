# ferrotorch — top-level justfile.
#
# Run `just --list` to discover recipes. Module-scoped recipes are
# listed via `just <module> --list` (e.g. `just ci-runner --list`).

# Self-hosted GitHub Actions runner ops. See `ci/runner/OPERATIONS.md`
# for the full deployment runbook; the recipes here automate the
# steps in that document.
#
# Module source lives at `ci/runner/mod.just`. Using the `.just` file
# form (not the directory form) for compatibility with older `just`
# versions that resolve directory-style modules differently.
mod ci-runner './ci/runner/mod.just'

# Convenience: default recipe shows the top-level help.
default:
    @just --list
