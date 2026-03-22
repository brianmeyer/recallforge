# RecallForge Release Checklist

RecallForge already has two release-facing GitHub Actions workflows:

- `ci.yml`: test matrix, distribution build, `twine check`, wheel smoke test, macOS import/backend checks, and HTTP server extra smoke coverage
- `publish.yml`: tag-triggered PyPI publish via trusted publishing (`v*`)

Use this checklist before cutting a version.

## 1. Version and surface audit

1. Update the version in both `pyproject.toml` and `src/recallforge/__init__.py`.
2. Review user-facing surfaces for drift:
   - `README.md`
   - `docs/ARCHITECTURE.md`
   - `docs/mcp-tools.md`
3. Confirm CLI help and MCP tool lists still match the docs.

## 2. Required local validation

Run these from the repo root:

```bash
pytest -q
bash tests/uat/test_install.sh
bash tests/uat/test_cli.sh
```

Those cover the unit/integration suite, fresh install path, CLI behavior, stdio server startup, and HTTP `/health` coverage when server extras are available.

## 3. Live validation on a capable host

If you have real models available on the machine, run:

```bash
UAT_MCP_LIVE=1 .venv/bin/python -m pytest -q tests/uat/test_uat_comprehensive.py -m integration -k real_backend -rs
UAT_MCP_LIVE=1 .venv/bin/python -m pytest -q tests/uat/test_uat_comprehensive.py -m integration -k external_client -rs
```

Then run the expanded benchmark:

```bash
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --output benchmarks/results/cross_modal_ablation_results.json
```

The benchmark now checkpoints to JSON as it runs. If the run is interrupted, the output file still contains partial results plus progress metadata.

## 4. Tag and publish

1. Commit the release changes.
2. Create the release tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

3. Watch the `Publish to PyPI` workflow complete successfully.

## 5. Post-release smoke check

Verify the released package from PyPI:

```bash
pip install "recallforge[mlx,server]"
recallforge --version
recallforge serve --http --mode embed --host 127.0.0.1 --port 7433
```

Then hit `http://127.0.0.1:7433/health` and confirm the process reports healthy model state.
