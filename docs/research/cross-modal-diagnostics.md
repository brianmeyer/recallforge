# Cross-Modal Retrieval Diagnostics

This report is generated from the saved cross-modal ablation JSON. It separates raw embedding alignment, derived-text contribution, reranker contribution, benchmark artifacts, and parent-memory scoring coverage.

## Source

- Benchmark: `cross_modal_ablation`
- Source version: `0.2.0`
- Source generated at: `2026-03-22T01:15:51.127774+00:00`
- Run status: `complete`
- Weak threshold: R@5 < 60.0%
- Query floor: 20 queries per category

## Weak And At-Risk Category Ranking

| Priority | Category | Queries | Best stage | Best R@5 | Vector R@5 | RRF R@5 | Hybrid R@5 | Key issues |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 0.85 | `image_to_document` | 2 | Vector-only | 0.0% | 0.0% | 0.0% | 0.0% | under_sampled_category, bm25_modality_blind, embedding_alignment_gap, derived_text_probe_lift |
| 0.85 | `video_to_document` | 1 | Vector-only | 0.0% | 0.0% | 0.0% | 0.0% | under_sampled_category, bm25_modality_blind, embedding_alignment_gap, derived_text_probe_insufficient |
| 0.80 | `video_to_image` | 2 | Vector-only | 0.0% | 0.0% | 0.0% | 0.0% | under_sampled_category, bm25_modality_blind, embedding_alignment_gap, derived_text_probe_insufficient |
| 0.47 | `video_to_text` | 3 | Vector-only | 33.3% | 33.3% | 33.3% | 33.3% | under_sampled_category, bm25_modality_blind, embedding_alignment_gap, derived_text_probe_insufficient |
| 0.20 | `image_to_image` | 3 | Vector-only | 100.0% | 100.0% | 100.0% | 100.0% | under_sampled_category, bm25_modality_blind, parent_asset_metrics_missing |
| 0.20 | `image_to_text` | 15 | Vector + BM25 (RRF) | 60.0% | 46.7% | 60.0% | 60.0% | under_sampled_category, bm25_modality_blind, embedding_alignment_gap, derived_text_probe_lift |
| 0.20 | `image_to_video` | 2 | Vector-only | 100.0% | 100.0% | 100.0% | 100.0% | under_sampled_category, bm25_modality_blind, parent_asset_metrics_missing, generic_query_artifact |
| 0.20 | `text_to_document` | 12 | Vector-only | 100.0% | 100.0% | 100.0% | 100.0% | under_sampled_category, parent_asset_metrics_missing, document_family_filter_gap |
| 0.20 | `video_to_video` | 1 | Vector-only | 100.0% | 100.0% | 100.0% | 100.0% | under_sampled_category, bm25_modality_blind, parent_asset_metrics_missing, media_query_path_missing |

## Diagnosis Summary

- `bm25_modality_blind`: 8
- `derived_text_probe_insufficient`: 3
- `derived_text_probe_lift`: 2
- `document_family_filter_gap`: 3
- `embedding_alignment_gap`: 5
- `generic_query_artifact`: 6
- `media_query_path_missing`: 4
- `parent_asset_metrics_missing`: 13
- `reranker_no_lift`: 4
- `under_sampled_category`: 11

## Prioritized Fix List

1. **search** - Add bounded cascaded media reranking only after cheap top-K retrieval. The weakest media-query categories are not rescued by current RRF/reranker stages, so REC-130 should target a strict top-K cascade instead of broad expensive scoring.
2. **evals** - Expand weak categories to at least 20 queries and keep parent-memory scoring. Several weak categories have 1-3 examples, which is too small to distinguish model weakness from benchmark noise; this maps directly to REC-160.
3. **indexing** - Represent document-family filters explicitly across pdf/docx/pptx roots and children. Document retrieval categories are evaluated without a proper document-family content filter, so unrelated images/videos can dominate media-query results.
4. **evals** - Replace placeholder media prompts with grounded intent labels and provenance. Queries such as 'related document' are useful smoke probes but too ambiguous for release-quality diagnostics.
5. **evals** - Rerun cross-modal ablation with the current harness to populate asset_level metrics. The checked-in result is from v0.2.0 and predates serialized asset-level rollups, so it cannot fully separate child-asset hits from parent-memory hits.
6. **ingest** - Keep strengthening captions, transcripts, and OCR as first-class retrieval text. Where RRF improves over vector-only, the improvement is evidence that derived text is helping and should be cached/versioned rather than recomputed ad hoc.
7. **model_research** - Benchmark visual/document-specialized retrievers against the weak categories. ViDoRe-style visual document retrieval and MTEB/BEIR-style qrels offer better external baselines for document-heavy failures than anecdotes from one synthetic corpus.

## Category Evidence

### `image_to_document`

- Queries: 2
- Target result family: `document_family`
- Configured benchmark content filter: `None`
- Best stage/R@5: Vector-only / 0.0%
- Raw vector R@5/R@10: 0.0% / 0.0%
- RRF R@5/R@10: 0.0% / 50.0%
- Hybrid R@5/R@10: 0.0% / 50.0%
- Top-5 audit source counts: `original_fts`=8, `original_vec`=8
- `under_sampled_category` (high): 2 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process image queries
- `embedding_alignment_gap` (high): Vector-only R@5=0.000; media-query categories need raw embedding alignment above 0.60.
- `derived_text_probe_lift` (positive): RRF improves R@5 by +0.000 and R@10 by +0.500 over raw vector search.
- `reranker_no_lift` (medium): Hybrid reranker changes R@5 by only +0.000 versus RRF.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `document_family_filter_gap` (medium): Benchmark cannot currently constrain results to the pdf/docx/pptx document family with a single content_type filter.
- `generic_query_artifact` (medium): Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.
- Example misses:
  - `related document` (image): expected ['documents/ai_strategy_report.docx', 'documents/ai_architecture_deck.pptx', 'documents/embedding_research.pdf']; top results `images/neural_network_diagram.png`, `images/whiteboard_brainstorm.png`, `images/whiteboard_architecture.png`
  - `related document` (image): expected ['documents/recallforge_spec.docx', 'documents/operations_manual.pdf']; top results `images/code_editor_screenshot.png`, `videos/coding_demo.mp4::transcript:0001@0.00s`, `images/whiteboard_architecture.png`

### `video_to_document`

- Queries: 1
- Target result family: `document_family`
- Configured benchmark content filter: `None`
- Best stage/R@5: Vector-only / 0.0%
- Raw vector R@5/R@10: 0.0% / 0.0%
- RRF R@5/R@10: 0.0% / 0.0%
- Hybrid R@5/R@10: 0.0% / 0.0%
- Top-5 audit source counts: `original_fts`=4, `original_vec`=4
- `under_sampled_category` (high): 1 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process video queries
- `embedding_alignment_gap` (high): Vector-only R@5=0.000; media-query categories need raw embedding alignment above 0.60.
- `derived_text_probe_insufficient` (high): RRF R@5=0.000 does not materially lift vector R@5=0.000.
- `reranker_no_lift` (medium): Hybrid reranker changes R@5 by only +0.000 versus RRF.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `document_family_filter_gap` (medium): Benchmark cannot currently constrain results to the pdf/docx/pptx document family with a single content_type filter.
- `generic_query_artifact` (medium): Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.
- `media_query_path_missing` (low): 1 per-query rows omit image_query_path or video_query_path instrumentation.
- Example misses:
  - `related document` (video): expected ['documents/ai_architecture_deck.pptx']; top results `videos/architecture_walkthrough.mp4`, `videos/architecture_walkthrough.mp4::frame:0004@15.00s`, `videos/architecture_walkthrough.mp4::frame:0006@25.00s`

### `video_to_image`

- Queries: 2
- Target result family: `image`
- Configured benchmark content filter: `image`
- Best stage/R@5: Vector-only / 0.0%
- Raw vector R@5/R@10: 0.0% / 100.0%
- RRF R@5/R@10: 0.0% / 100.0%
- Hybrid R@5/R@10: 0.0% / 100.0%
- Top-5 audit source counts: `original_fts`=7, `original_vec`=8
- `under_sampled_category` (high): 2 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process video queries
- `embedding_alignment_gap` (high): Vector-only R@5=0.000; media-query categories need raw embedding alignment above 0.60.
- `derived_text_probe_insufficient` (high): RRF R@5=0.000 does not materially lift vector R@5=0.000.
- `reranker_no_lift` (medium): Hybrid reranker changes R@5 by only +0.000 versus RRF.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `generic_query_artifact` (medium): Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.
- `media_query_path_missing` (low): 2 per-query rows omit image_query_path or video_query_path instrumentation.
- Example misses:
  - `related image` (video): expected ['images/forest_landscape.png', 'images/mountain_landscape.png', 'images/ocean_beach.png']; top results `videos/nature_timelapse.mp4::frame:0006@25.00s`, `videos/nature_timelapse.mp4::frame:0005@20.00s`, `videos/nature_timelapse.mp4::frame:0007@30.00s`
  - `related image` (video): expected ['images/whiteboard_brainstorm.png', 'images/whiteboard_architecture.png']; top results `videos/whiteboard_session.mp4::frame:0003@10.00s`, `videos/whiteboard_session.mp4::frame:0002@5.00s`, `videos/whiteboard_session.mp4::frame:0001@0.00s`

### `video_to_text`

- Queries: 3
- Target result family: `text`
- Configured benchmark content filter: `text`
- Best stage/R@5: Vector-only / 33.3%
- Raw vector R@5/R@10: 33.3% / 33.3%
- RRF R@5/R@10: 33.3% / 33.3%
- Hybrid R@5/R@10: 33.3% / 33.3%
- Top-5 audit source counts: `original_fts`=8, `original_vec`=12
- `under_sampled_category` (high): 3 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process video queries
- `embedding_alignment_gap` (high): Vector-only R@5=0.333; media-query categories need raw embedding alignment above 0.60.
- `derived_text_probe_insufficient` (high): RRF R@5=0.333 does not materially lift vector R@5=0.333.
- `reranker_no_lift` (medium): Hybrid reranker changes R@5 by only +0.000 versus RRF.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `generic_query_artifact` (medium): Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.
- `media_query_path_missing` (low): 3 per-query rows omit image_query_path or video_query_path instrumentation.
- Example misses:
  - `related text` (video): expected ['text/tech_cybersecurity.md', 'text/tech_cloud_computing.md']; top results `videos/coding_demo.mp4::transcript:0002@2.00s`, `videos/whiteboard_session.mp4::transcript:0001@0.00s`, `text/medicine_nutrition.md`
  - `related text` (video): expected ['text/architecture_gothic.md', 'text/architecture_modern.md', 'text/architecture_blueprints.md']; top results `videos/coding_demo.mp4::transcript:0002@2.00s`, `text/ai_agents.md`, `documents/ai_architecture_deck.pptx::slide:0001`

### `image_to_image`

- Queries: 3
- Target result family: `image`
- Configured benchmark content filter: `image`
- Best stage/R@5: Vector-only / 100.0%
- Raw vector R@5/R@10: 100.0% / 100.0%
- RRF R@5/R@10: 100.0% / 100.0%
- Hybrid R@5/R@10: 100.0% / 100.0%
- Top-5 audit source counts: `original_fts`=8, `original_vec`=12
- `under_sampled_category` (high): 3 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process image queries
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.

### `image_to_text`

- Queries: 15
- Target result family: `text`
- Configured benchmark content filter: `text`
- Best stage/R@5: Vector + BM25 (RRF) / 60.0%
- Raw vector R@5/R@10: 46.7% / 73.3%
- RRF R@5/R@10: 60.0% / 86.7%
- Hybrid R@5/R@10: 60.0% / 86.7%
- Top-5 audit source counts: `original_fts`=60, `original_vec`=63
- `under_sampled_category` (medium): 15 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process image queries
- `embedding_alignment_gap` (high): Vector-only R@5=0.467; media-query categories need raw embedding alignment above 0.60.
- `derived_text_probe_lift` (positive): RRF improves R@5 by +0.133 and R@10 by +0.133 over raw vector search.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `generic_query_artifact` (medium): Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.
- Example misses:
  - `<empty media query>` (image): expected ['text/nature_oceans.md']; top results `text/sports_golf.md`, `text/music_production.md`, `text/nature_forests.md`
  - `<empty media query>` (image): expected ['text/ai_agents.md', 'text/tech_edge_ai.md']; top results `videos/architecture_walkthrough.mp4::transcript:0001@0.00s`, `text/tech_cloud_computing.md`, `videos/whiteboard_session.mp4::transcript:0001@0.00s`
  - `<empty media query>` (image): expected ['text/ai_agents.md', 'text/tech_cloud_computing.md']; top results `videos/whiteboard_session.mp4::transcript:0002@2.00s`, `videos/whiteboard_session.mp4::transcript:0001@0.00s`, `videos/architecture_walkthrough.mp4::transcript:0001@0.00s`

### `image_to_video`

- Queries: 2
- Target result family: `video`
- Configured benchmark content filter: `video`
- Best stage/R@5: Vector-only / 100.0%
- Raw vector R@5/R@10: 100.0% / 100.0%
- RRF R@5/R@10: 100.0% / 100.0%
- Hybrid R@5/R@10: 100.0% / 100.0%
- Top-5 audit source counts: `original_fts`=2, `original_vec`=8
- `under_sampled_category` (high): 2 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process image queries
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `generic_query_artifact` (medium): Most media-query prompts are generic placeholders such as 'related document', so scores mix retrieval quality with query-definition ambiguity.

### `text_to_document`

- Queries: 12
- Target result family: `document_family`
- Configured benchmark content filter: `None`
- Best stage/R@5: Vector-only / 100.0%
- Raw vector R@5/R@10: 100.0% / 100.0%
- RRF R@5/R@10: 100.0% / 100.0%
- Hybrid R@5/R@10: 100.0% / 100.0%
- Top-5 audit source counts: `original_fts`=44, `original_vec`=48
- `under_sampled_category` (medium): 12 queries is below the 20-query diagnostic floor.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `document_family_filter_gap` (medium): Benchmark cannot currently constrain results to the pdf/docx/pptx document family with a single content_type filter.

### `video_to_video`

- Queries: 1
- Target result family: `video`
- Configured benchmark content filter: `video`
- Best stage/R@5: Vector-only / 100.0%
- Raw vector R@5/R@10: 100.0% / 100.0%
- RRF R@5/R@10: 100.0% / 100.0%
- Hybrid R@5/R@10: 100.0% / 100.0%
- Top-5 audit source counts: `original_vec`=4
- `under_sampled_category` (high): 1 queries is below the 20-query diagnostic floor.
- `bm25_modality_blind` (medium): BM25 can't process video queries
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- `media_query_path_missing` (low): 1 per-query rows omit image_query_path or video_query_path instrumentation.

### `text_to_image`

- Queries: 18
- Target result family: `image`
- Configured benchmark content filter: `image`
- Best stage/R@5: Vector + BM25 (RRF) / 94.4%
- Raw vector R@5/R@10: 88.9% / 94.4%
- RRF R@5/R@10: 94.4% / 94.4%
- Hybrid R@5/R@10: 94.4% / 94.4%
- Top-5 audit source counts: `original_fts`=17, `original_vec`=72
- `under_sampled_category` (medium): 18 queries is below the 20-query diagnostic floor.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- Example misses:
  - `coastal landscape photography` (text): expected ['images/ocean_beach.png']; top results `videos/nature_timelapse.mp4::frame:0006@25.00s`, `videos/nature_timelapse.mp4::frame:0005@20.00s`, `videos/nature_timelapse.mp4::frame:0003@10.00s`

### `text_to_video`

- Queries: 15
- Target result family: `video`
- Configured benchmark content filter: `video`
- Best stage/R@5: Vector-only / 100.0%
- Raw vector R@5/R@10: 100.0% / 100.0%
- RRF R@5/R@10: 100.0% / 100.0%
- Hybrid R@5/R@10: 100.0% / 100.0%
- Top-5 audit source counts: `original_fts`=1, `original_vec`=60
- `under_sampled_category` (medium): 15 queries is below the 20-query diagnostic floor.
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.

### `mixed_modal`

- Queries: 20
- Target result family: `mixed`
- Configured benchmark content filter: `None`
- Best stage/R@5: BM25-only / 95.0%
- Raw vector R@5/R@10: 85.0% / 100.0%
- RRF R@5/R@10: 95.0% / 100.0%
- Hybrid R@5/R@10: 90.0% / 100.0%
- Top-5 audit source counts: `original_fts`=75, `original_vec`=78
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- Example misses:
  - `comprehensive guide to athletic performance` (text): expected ['text/sports_running.md', 'text/sports_cycling.md', 'text/sports_swimming.md', 'text/sports_yoga.md', 'text/medicine_nutrition.md', 'text/medicine_cardiology.md']; top results `text/cooking_sourdough.md`, `documents/ai_architecture_deck.pptx::slide:0003`, `documents/edge_deployment_guide.pdf::page:0001`

### `text_to_text`

- Queries: 60
- Target result family: `text`
- Configured benchmark content filter: `text`
- Best stage/R@5: Vector + BM25 (RRF) / 90.0%
- Raw vector R@5/R@10: 88.3% / 90.0%
- RRF R@5/R@10: 90.0% / 91.7%
- Hybrid R@5/R@10: 90.0% / 91.7%
- Top-5 audit source counts: `original_fts`=188, `original_vec`=241
- `parent_asset_metrics_missing` (medium): Saved payload lacks asset_level metrics; rerun the benchmark with the current harness to separate parent-memory and child-asset hits.
- Example misses:
  - `how do computers understand the meaning of words` (text): expected ['text/ai_embeddings.md']; top results `text/ai_transformers.md`, `text/tech_quantum_computing.md`, `text/tech_cloud_computing.md`
  - `underwater basket weaving techniques` (text): expected []; top results `text/sports_swimming.md`, `text/architecture_gothic.md`, `text/cooking_sourdough.md`
  - `medieval jousting tournament rules and equipment` (text): expected []; top results `text/sports_swimming.md`, `text/sports_cycling.md`, `text/history_renaissance.md`

## Method Notes

- Vector-only is treated as the raw embedding baseline.
- RRF lift over vector-only is treated as evidence from derived text probes such as captions, transcripts, OCR, or BM25 text.
- Hybrid-minus-RRF isolates the current reranker contribution.
- Parent-memory versus asset-level scoring is only available when the source payload includes asset_level metrics.

## External Evaluation References

- [BEIR](https://github.com/beir-cellar/beir) structures retrieval evaluation around corpus, queries, qrels, run results, and metrics such as NDCG, MAP, Recall, Precision, and MRR.
- [MTEB](https://github.com/embeddings-benchmark/mteb) is the broader embedding and retrieval evaluation framework now used by ViDoRe for single-model retriever submissions.
- [ViDoRe pipeline evaluation](https://github.com/illuin-tech/vidore-benchmark) explicitly covers multi-stage, hybrid, reranking, OCR, and custom preprocessing pipelines for visual document retrieval.
