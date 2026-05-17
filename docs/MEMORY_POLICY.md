# RecallForge Memory Policy

This document explains how RecallForge scopes, stores, and ranks memories.

## Scope

Every memory can be filtered by:

- `collection`
- `user_id`
- `session_id`
- `project_id`
- `profile`

Search, list, get, update, and delete paths must use the same namespace values that were used during ingest. This keeps unrelated users, sessions, projects, and profiles isolated inside one local store.

## Memory Shape

RecallForge stores complex files as a root memory plus child assets:

- Text memory: one root memory with text chunks.
- Image memory: one root image memory, with generated caption text and tags when captioning is enabled.
- Video memory: one root video memory, with frame children and transcript children when available.
- Document memory: one root document memory, with section/page children and OCR siblings when available.
- Audio memory: one root audio memory, with timestamped transcript children from sidecar transcripts.
- Conversation memory: one root text memory with a summary/participant overview, plus one child memory per turn.

The root memory uses `memory_role="root"` and child assets use `memory_role="child"` with `memory_root_path` pointing back to the root.

## Ranking

RecallForge combines BM25 and vector search through RRF, then reranks in `hybrid` mode. Before fusion, storage search applies memory policy:

- Expired rows are filtered out with `expires_at`.
- `importance` can add up to a 15% boost before score normalization.
- Fresh rows receive a small recency boost.
- Tags are carried into results and memory rollups for agent context.

The boost is intentionally modest: it can break ties and surface important recent memories, but it does not override strong semantic or keyword evidence.

## Audio

Audio support is transcript-first. Put a `.srt`, `.vtt`, `.txt`, or `.transcript.json` sidecar next to the audio file. RecallForge indexes the audio as a root `content_type="audio"` memory and indexes transcript segments as text child memories.

Raw audio transcription and dedicated audio encoders are not part of the shipped v0.3.0 policy.

## Conversations

Use `memory_add_conversation` when an agent or app wants to persist a thread. RecallForge stores the parent at the supplied `path` and stores each turn at `path::turn:0001`, `path::turn:0002`, and so on. All turns share the parent `memory_id`, so if several turns match a query, search and explanation output roll them up into the parent conversation with evidence paths.

## Memory Graph

Every indexed text-bearing evidence unit can also produce lightweight entity mentions and co-mention relation edges. That includes normal text memories, OCR/document sections, transcripts, captions, and conversation turns. Graph rows store `memory_id`, `memory_root_path`, `file_path`, and a short evidence snippet, so same-entity navigation remains traceable to the source memory.

Use `memory_graph_entities` to inspect entities for a memory, path, or entity key. Use `memory_graph_related` to find other memories that share extracted entities with a seed memory or entity. This graph enrichment is intentionally local and deterministic: it adds navigation and grouping without introducing an external NLP service.
