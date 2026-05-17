# RecallForge Runtime Budgets

These are the release-facing budgets for local runs on Apple Silicon with MLX 4-bit.

## Budgets

| Path | Budget |
|------|--------|
| Warm embed search p50 | <= 60 ms |
| Warm embed search p95 | <= 100 ms |
| Safe MLX smoke RSS | <= 6144 MB |
| MCP blocking operations | bounded by `RECALLFORGE_MCP_MAX_CONCURRENCY` |
| Heavy MLX multimodal operations | default serialized by `RECALLFORGE_MLX_HEAVY_OP_CONCURRENCY=1` |
| Captioner idle retention | 30 seconds by default, via `RECALLFORGE_CAPTIONER_IDLE_SECONDS` |

## Validation

Run the fast local checks first:

```bash
pytest -q
bash tests/uat/test_install.sh
bash tests/uat/test_cli.sh
```

On a capable MLX host, run the bounded benchmark lane:

```bash
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --smoke-profile safe --rss-limit-mb 6144
```

Then run the full profile only after the safe smoke completes:

```bash
.venv/bin/python benchmarks/cross_modal_ablation.py --backend mlx --expansion-profile caption_only --output benchmarks/results/cross_modal_ablation_results.json
```

The benchmark JSON records `telemetry.peak_rss_mb`, `configuration.rss_limit_mb`, and `configuration.smoke_profile`.
