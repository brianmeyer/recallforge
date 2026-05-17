# RecallForge Alpha and Beta Testing Program

This program keeps early testing useful without turning a local-first memory tool into silent telemetry. Alpha and beta users should share feedback intentionally through GitHub Discussions or Issues.

## Goals

- Recruit 5-10 alpha users who use RecallForge with real local notes, documents, images, audio, or videos.
- Validate install, MCP setup, ingest, search, memory rollups, and local runtime safety.
- Move successful alpha workflows into a broader beta with structured feedback.
- Keep crash reporting opt-in, inspectable, and manually shared.

## Channels

- Alpha feedback: `https://github.com/brianmeyer/recallforge/discussions`
- Bug reports: `https://github.com/brianmeyer/recallforge/issues/new/choose`
- Security issues: use the repository security policy or private maintainer contact, not public Discussions.

GitHub discussion category forms live under `.github/DISCUSSION_TEMPLATE/`. To activate them, enable GitHub Discussions and create categories whose slugs match the template filenames, such as `alpha-feedback` and `beta-feedback`.

## Tester Cohorts

Alpha users should cover:

- Apple Silicon local agents using MLX
- CPU/CUDA users through the torch backend
- Claude Desktop or another MCP host
- text-heavy personal notes
- multimodal folders with images, PDFs, audio transcripts, and short videos

Beta users should add:

- larger folders and watch-folder workflows
- repeated reindexing
- HTTP/SSE MCP clients
- cross-modal query workflows
- release-candidate install checks from PyPI

## Feature Flags

Run this to see the supported feature flags and current values:

```bash
recallforge flags
recallforge flags --json
```

Recommended alpha defaults:

- Keep `RECALLFORGE_ENABLE_MEDIA_RERANKING=0` unless a tester is explicitly validating capped media reranking.
- Keep `RECALLFORGE_ENABLE_RAW_VIDEO_QUERY_EMBEDDING=0` unless validating raw video-query behavior.
- Keep `RECALLFORGE_ENABLE_MLX_NATIVE_VIDEO_PROCESSING=0` unless validating native MLX video decoding.
- Use `RECALLFORGE_TRACE=1` only while collecting a diagnostic reproduction.

The canonical environment variable reference is `docs/ENV_VARS.md`.

## Opt-In Crash Reports

RecallForge does not send crash reports automatically. A tester can create a local JSON report and review it before attaching it to a Discussion or Issue:

```bash
recallforge crash-report --output recallforge-crash-report.json --message "Search crashed after a video query"
```

To include allowlisted `RECALLFORGE_*` values with home paths redacted:

```bash
recallforge crash-report --include-env --output recallforge-crash-report.json
```

Crash reports include:

- RecallForge version
- Python implementation/version
- OS family, release, and machine architecture
- effective feature flag values
- optional allowlisted `RECALLFORGE_*` environment values
- a user-provided message

Crash reports do not include:

- indexed content
- search queries
- arbitrary environment variables
- automatic network upload
- API keys or tokens unless a user manually adds them after generation

## Alpha Checklist

1. Install from PyPI or the current release branch.
2. Run `recallforge --version`.
3. Run `recallforge flags --json` and attach the output if testing experimental flags.
4. Start MCP with `recallforge serve --mode embed`.
5. Ingest a tiny folder with text plus one media file.
6. Search text to text, text to media, and media to text.
7. Reindex the same folder and confirm old/stale results do not appear.
8. Share feedback through the alpha discussion template.

## Beta Checklist

1. Install from the release candidate wheel or PyPI package.
2. Run the MCP server through the real client host used day to day.
3. Ingest a representative local folder.
4. Run at least five saved workflows:
   - exact lookup
   - broad semantic lookup
   - image query
   - video or transcript-backed query
   - conversation or memory rollup lookup
5. Test one runtime feature flag intentionally, then return to default.
6. Attach an opt-in crash report only if the run fails.
7. Record quality and latency notes in the beta discussion template.

## Exit Criteria

Alpha is complete when:

- at least five users complete the alpha checklist
- no release-blocking install or MCP startup issue remains open
- crash reports, if any, are reproducible or explicitly accepted as known limitations

Beta is complete when:

- at least ten workflow reports are collected across at least three machine profiles
- PyPI install and MCP startup are boring
- docs explain the safest defaults and opt-in flags
- known limitations are documented before release
