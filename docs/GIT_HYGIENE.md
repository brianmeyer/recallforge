# RecallForge Git Hygiene

Use this checklist after releases, merged PRs, and long agent sessions.

## Current Repository Policy

- Default branch: `master`
- Release tags: `vX.Y.Z`
- Agent branches: `codex/<short-description>`
- GitHub merged-head branch cleanup: enabled

GitHub automatically deletes PR head branches after merge. Local cleanup is still useful because remote-tracking refs, generated build artifacts, and extra worktrees can hang around on a developer machine.

## Routine Cleanup

Check repository state:

```bash
git status --short --branch
git fetch --prune
git branch --all --verbose --no-abbrev
git worktree list --porcelain
gh pr list --repo brianmeyer/recallforge --state open --limit 20
gh run list --repo brianmeyer/recallforge --limit 10
```

Remove local merged branches after confirming they are no longer needed:

```bash
git switch master
git pull --ff-only recallforge master
git branch --merged master
git branch -d codex/<merged-branch>
```

Remove ignored local build/test artifacts:

```bash
rm -rf build dist src/recallforge.egg-info .pytest_cache
```

Do not remove unmerged branches, alternate worktrees, or user-created files unless they are clearly part of the cleanup task.

## Release Cleanup

After a release tag is pushed:

```bash
gh run list --repo brianmeyer/recallforge --workflow publish.yml --limit 5
python -m pip index versions recallforge
python -m pip install "recallforge==X.Y.Z"
```

Then verify:

- `master` is clean and current with `recallforge/master`
- the release tag points at the intended merge commit
- no release PRs are left open
- PyPI shows the new version as latest
