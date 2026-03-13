#!/usr/bin/env bash
# test_document_ingest.sh - Document ingest UAT.
# Validates local-first PDF/DOCX/PPTX extraction through the shipped CLI path.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Document Ingest Tests"

cd "$REPO_ROOT"
trap cleanup_store EXIT

USE_LIVE_BACKEND=1
SELECTED_BACKEND="$(select_live_backend || true)"
if [[ -z "${SELECTED_BACKEND}" ]]; then
    USE_LIVE_BACKEND=0
    info "No usable live backend on this host; using deterministic CLI harness backend."
fi

export RECALLFORGE_BACKEND="${SELECTED_BACKEND:-torch}"
if [[ "${SELECTED_BACKEND:-}" == "mlx" ]]; then
    export RECALLFORGE_MLX_QUANTIZE=4bit
fi
export RECALLFORGE_MODE=embed
export RECALLFORGE_STORE_PATH="${UAT_STORE}"

run_cli() {
    local args_json
    args_json=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")
    UAT_CLI_ARGS_JSON="$args_json" UAT_CLI_USE_LIVE="$USE_LIVE_BACKEND" python3 <<'PYEOF'
import contextlib
import io
import json
import os
import re
import sys

sys.path.insert(0, "src")

import recallforge
from recallforge.backends.base import BackendInfo
from recallforge.cli import main


class HarnessBackend:
    def __init__(self):
        self._mode = os.environ.get("RECALLFORGE_MODE", "embed")

    def set_mode(self, mode):
        self._mode = mode

    def get_mode(self):
        return self._mode

    def _vec(self, seed: str):
        import hashlib

        values = [0.0] * 2048
        tokens = re.findall(r"[a-z0-9]+", seed.lower())
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).hexdigest()
            values[int(digest[:8], 16) % 2048] += 1.0
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed_text(self, text):
        return self._vec(text)

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]

    def embed_image(self, path):
        return self._vec(os.path.basename(path))

    def rerank(self, query, documents):
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores = []
        for doc in documents:
            doc_tokens = set(re.findall(r"[a-z0-9]+", doc.get("text", "").lower()))
            overlap = len(query_tokens & doc_tokens)
            scores.append(float(overlap) / float(max(len(query_tokens), 1)))
        return scores

    def expand_query(self, query):
        return {"lex": query, "vec": query, "hyde": query}

    def needs_expander(self):
        return False

    def needs_reranker(self):
        return False

    def get_info(self):
        return BackendInfo(
            name="harness",
            device="cpu",
            dtype="float32",
            embedder_loaded=True,
            reranker_loaded=False,
            expander_loaded=False,
            memory_allocated_gb=0.0,
            quantization="none",
        )


if os.environ.get("UAT_CLI_USE_LIVE") != "1":
    real_get_storage = recallforge.get_storage
    backend = HarnessBackend()
    recallforge.get_backend = lambda: backend
    recallforge.get_storage = lambda store_path=None: real_get_storage(store_path)

argv = json.loads(os.environ["UAT_CLI_ARGS_JSON"])
buffer = io.StringIO()
exit_code = 0
with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
    sys.argv = ["recallforge", *argv]
    exit_code = main()
print(buffer.getvalue(), end="")
raise SystemExit(exit_code)
PYEOF
}

subsection "Generate Document Fixtures"

DOC_META=$(python3 "${HELPERS_DIR}/generate_test_documents.py" "${UAT_STORE}/document_fixtures")
DOCX_PATH=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["files"]["docx"])' <<<"$DOC_META")
PPTX_PATH=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["files"]["pptx"])' <<<"$DOC_META")
PDF_PATH=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["files"]["pdf"])' <<<"$DOC_META")

for file_path in "$DOCX_PATH" "$PPTX_PATH" "$PDF_PATH"; do
    if [[ -f "$file_path" ]]; then
        pass "fixture created: $(basename "$file_path")"
    else
        fail "fixture created: $(basename "$file_path")"
    fi
done

subsection "CLI Document Index"

for file_path in "$DOCX_PATH" "$PPTX_PATH" "$PDF_PATH"; do
    OUTPUT=$(run_cli index "$file_path" --collection doc_cli_test --store-path "$UAT_STORE" 2>&1 || true)
    if echo "$OUTPUT" | grep -q "Indexing document" && ! echo "$OUTPUT" | grep -qE "Traceback|Error indexing|ValueError:"; then
        pass "recallforge index <$(basename "$file_path")>"
    else
        fail "recallforge index <$(basename "$file_path")>"
        echo "    Output: $(echo "$OUTPUT" | head -5)"
    fi
done

subsection "Text Search Over Extracted Sections"

OUTPUT=$(run_cli search "quarterly planning embeddings reranking local-first MCP deployment" \
    --collection doc_cli_test \
    --content-type text \
    --store-path "$UAT_STORE" 2>&1 || true)
if echo "$OUTPUT" | grep -q "planning_notes.docx::section:"; then
    pass "DOCX-derived sections searchable via CLI"
else
    fail "DOCX-derived sections searchable via CLI"
    echo "    Output: $(echo "$OUTPUT" | head -8)"
fi

OUTPUT=$(run_cli search "deployment checklist" \
    --collection doc_cli_test \
    --content-type text \
    --store-path "$UAT_STORE" 2>&1 || true)
if echo "$OUTPUT" | grep -q "deployment_review.pptx::slide:"; then
    pass "PPTX-derived slides searchable via CLI"
else
    fail "PPTX-derived slides searchable via CLI"
    echo "    Output: $(echo "$OUTPUT" | head -8)"
fi

OUTPUT=$(run_cli search "local-first PDF notes" \
    --collection doc_cli_test \
    --content-type text \
    --store-path "$UAT_STORE" 2>&1 || true)
if echo "$OUTPUT" | grep -q "mcp_overview.pdf::page:"; then
    pass "PDF-derived pages searchable via CLI"
else
    fail "PDF-derived pages searchable via CLI"
    echo "    Output: $(echo "$OUTPUT" | head -8)"
fi

subsection "Derived Asset Inspection"

python3 <<PYEOF
import sys
sys.path.insert(0, "src")
from recallforge import get_storage

store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'doc_cli_test'").to_list()
docx_rows = [r for r in rows if "planning_notes.docx::section:" in r.get("file_path", "")]
pptx_rows = [r for r in rows if "deployment_review.pptx::slide:" in r.get("file_path", "")]
pdf_rows = [r for r in rows if "mcp_overview.pdf::page:" in r.get("file_path", "")]
print(f"DOCX_ROWS={len(docx_rows)}")
print(f"PPTX_ROWS={len(pptx_rows)}")
print(f"PDF_ROWS={len(pdf_rows)}")
PYEOF

DOCX_ROWS=$(python3 - <<PYEOF
import sys
sys.path.insert(0, "src")
from recallforge import get_storage
store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'doc_cli_test'").to_list()
print(sum(1 for r in rows if "planning_notes.docx::section:" in r.get("file_path", "")))
PYEOF
)

PPTX_ROWS=$(python3 - <<PYEOF
import sys
sys.path.insert(0, "src")
from recallforge import get_storage
store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'doc_cli_test'").to_list()
print(sum(1 for r in rows if "deployment_review.pptx::slide:" in r.get("file_path", "")))
PYEOF
)

PDF_ROWS=$(python3 - <<PYEOF
import sys
sys.path.insert(0, "src")
from recallforge import get_storage
store = get_storage("${UAT_STORE}")
rows = store._embeddings_table.search().where("collection = 'doc_cli_test'").to_list()
print(sum(1 for r in rows if "mcp_overview.pdf::page:" in r.get("file_path", "")))
PYEOF
)

if [[ "$DOCX_ROWS" -ge 1 ]]; then
    pass "DOCX ingest created structured embeddings ($DOCX_ROWS)"
else
    fail "DOCX ingest created structured embeddings"
fi

if [[ "$PPTX_ROWS" -ge 1 ]]; then
    pass "PPTX ingest created structured embeddings ($PPTX_ROWS)"
else
    fail "PPTX ingest created structured embeddings"
fi

if [[ "$PDF_ROWS" -ge 1 ]]; then
    pass "PDF ingest created structured embeddings ($PDF_ROWS)"
else
    fail "PDF ingest created structured embeddings"
fi

print_summary "Document Ingest Tests"
