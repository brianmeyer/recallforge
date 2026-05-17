# RecallForge Release Checklist

RecallForge already has two release-facing GitHub Actions workflows:

- `ci.yml`: test matrix, distribution build, `twine check`, wheel smoke test, macOS import/backend checks, HTTP server extra smoke coverage, and current Node 24 action pins
- `publish.yml`: tag-triggered PyPI publish via trusted publishing (`v*`) with current Node 24 action pins

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
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --expansion-profile caption_only --output benchmarks/results/cross_modal_ablation_results.json
```

The benchmark now checkpoints to JSON as it runs. If the run is interrupted, the output file still contains partial results plus progress metadata.

For safer local validation after the MLX hardening work, prefer the bounded smoke lane first:

```bash
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --smoke-profile safe --expansion-profile caption_only
```

That profile defaults to a smaller stage/query footprint and can enforce an RSS stop condition:

```bash
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --smoke-profile safe --rss-limit-mb 6144
```

The output JSON now records:

- `configuration.smoke_profile`
- `configuration.rss_limit_mb`
- `telemetry.peak_rss_mb`

Use the full benchmark only after the safe smoke completes cleanly on the target machine.

For query-expansion release decisions, compare at least these profiles:

```bash
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --expansion-profile caption_only
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --expansion-profile heuristic
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --expansion-profile qwen
```

Profile meanings:

- `caption_only`: shipped default baseline for media queries. Text queries do not expand; image/video queries still use caption or transcript BM25 probes.
- `heuristic`: opt-in expansion branches using the legacy heuristic rewrite fallback.
- `qwen`: opt-in expansion branches using the backend `generate_text()` path when available.
- `off`: pure no-expansion baseline, including no media caption probe, useful for measuring the value of caption/transcript query text itself.

When you omit `--output`, the benchmark now keeps profile-specific filenames for non-default runs, for example `cross_modal_ablation_results_qwen.json`.

### MLX safety notes

- MLX heavy multimodal operations are now intentionally serialized by default via `RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY=1`.
- On MLX, raw video query embedding is no longer the default hot path. RecallForge prefers caption/transcript-first retrieval unless `RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING=1` is set.
- On MLX, qwen-vl-utils native video decoding is now also opt-in. RecallForge defaults to frame/caption fallbacks unless `RECALLFORGE_ENABLE_MLX_NATIVE_VIDEO_PROCESSING=1` is set.
- If you do opt back into native MLX video decoding, prefer `FORCE_QWENVL_VIDEO_READER=torchcodec` per Qwen's upstream guidance.
- Direct image/video indexing and query expansion now schedule an MLX captioner idle unload. Tune with `RECALLFORGE_CAPTIONER_IDLE_SECONDS`; batch ingest still unloads the captioner immediately after the batch.
- The raw-video path now has explicit frame and pixel budget knobs:
  - `RECALLFORGE_MLX_VIDEO_SAMPLE_FPS`
  - `RECALLFORGE_MLX_VIDEO_MAX_FRAMES`
  - `RECALLFORGE_MLX_VIDEO_FALLBACK_MAX_FRAMES`
  - `RECALLFORGE_MLX_MIN_PIXELS`
  - `RECALLFORGE_MLX_MAX_PIXELS`

### Audio release notes

- Audio ingest is transcript-first. A `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`, or similar audio file must have a sibling `.srt`, `.vtt`, `.txt`, or `.transcript.json` sidecar.
- `ingest`, `index_audio`, watch-folder indexing, CLI indexing, and `content_type="audio"` filters cover the shipped audio path.
- Dedicated raw-audio embeddings or transcription are still future work; release checks should verify sidecar transcript retrieval rather than microphone/audio decoding.

## 4. Tag and publish

1. Commit the release changes.
2. Create the release tag:

```bash
git tag vX.Y.Z
git push recallforge vX.Y.Z
```

If your checkout uses a conventional `origin` remote instead of `recallforge`, push the tag to `origin`.

3. Watch the `Publish to PyPI` workflow complete successfully:

```bash
gh run list --repo brianmeyer/recallforge --workflow publish.yml --limit 5
```

## 5. Post-release smoke check

Verify the released package from PyPI:

```bash
pip install "recallforge[mlx,server]"
recallforge --version
recallforge serve --http --mode embed --host 127.0.0.1 --port 7433
```

Then hit `http://127.0.0.1:7433/health` and confirm the process reports healthy model state.

## 6. Git cleanup

After the release PR is merged and the tag is published, follow the routine in [`docs/GIT_HYGIENE.md`](GIT_HYGIENE.md) to prune remote-tracking refs, remove local merged branches, and clear generated build artifacts.
