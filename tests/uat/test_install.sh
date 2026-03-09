#!/usr/bin/env bash
# test_install.sh - RecallForge installation UAT.
# Tests fresh venv creation, pip install, CLI entry points, and imports.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/helpers/common.sh"

section "RecallForge Install Tests"
VENV_BASE="${TMPDIR:-/tmp}/recallforge-install-test-$$"
trap "rm -rf '${VENV_BASE}'" EXIT

# ──────────────────────────────────────────────
subsection "Base Install (PyTorch)"
# ──────────────────────────────────────────────
VENV1="${VENV_BASE}/base"
python3 -m venv "$VENV1"
source "${VENV1}/bin/activate"

info "Installing recallforge (base) from ${REPO_ROOT} ..."
pip install -q "${REPO_ROOT}" 2>&1 | tail -1

# CLI entry point
if recallforge --help >/dev/null 2>&1; then
    pass "recallforge --help works"
else
    fail "recallforge --help failed"
fi

# Version flag
VERSION_OUT=$(recallforge --version 2>&1)
if echo "$VERSION_OUT" | grep -q "RecallForge"; then
    pass "recallforge --version prints version (${VERSION_OUT})"
else
    fail "recallforge --version unexpected output: ${VERSION_OUT}"
fi

# Python import
if python3 -c "import recallforge; print(recallforge.__version__)" >/dev/null 2>&1; then
    pass "import recallforge works"
else
    fail "import recallforge failed"
fi

# Check core deps are installed
for dep in lancedb pyarrow PIL numpy mcp transformers torch scipy; do
    mod="$dep"
    [[ "$dep" == "PIL" ]] && mod="PIL.Image"
    if python3 -c "import $mod" 2>/dev/null; then
        pass "Dependency present: $dep"
    else
        fail "Missing dependency: $dep"
    fi
done

# Submodule imports
for mod in recallforge.backends recallforge.storage recallforge.search recallforge.server recallforge.cli; do
    if python3 -c "import $mod" 2>/dev/null; then
        pass "Import: $mod"
    else
        fail "Import failed: $mod"
    fi
done

deactivate

# ──────────────────────────────────────────────
subsection "MLX Install (Apple Silicon)"
# ──────────────────────────────────────────────
if is_apple_silicon; then
    VENV2="${VENV_BASE}/mlx"
    python3 -m venv "$VENV2"
    source "${VENV2}/bin/activate"

    info "Installing recallforge[mlx] ..."
    pip install -q "${REPO_ROOT}[mlx]" 2>&1 | tail -1

    if python3 -c "import mlx" 2>/dev/null; then
        pass "MLX package installed"
    else
        fail "MLX package missing after install"
    fi

    if python3 -c "from recallforge.backends import MLX_AVAILABLE; assert MLX_AVAILABLE" 2>/dev/null; then
        pass "MLX_AVAILABLE is True"
    else
        fail "MLX_AVAILABLE is False despite mlx install"
    fi

    if recallforge --help >/dev/null 2>&1; then
        pass "recallforge CLI works with MLX install"
    else
        fail "recallforge CLI broken with MLX install"
    fi

    deactivate
else
    skip "MLX install (not Apple Silicon)"
fi

# ──────────────────────────────────────────────
subsection "CUDA Install"
# ──────────────────────────────────────────────
if has_cuda; then
    VENV3="${VENV_BASE}/cuda"
    python3 -m venv "$VENV3"
    source "${VENV3}/bin/activate"

    info "Installing recallforge[cuda] ..."
    pip install -q "${REPO_ROOT}[cuda]" 2>&1 | tail -1

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        pass "CUDA available after install"
    else
        fail "CUDA not available after install"
    fi

    if recallforge --help >/dev/null 2>&1; then
        pass "recallforge CLI works with CUDA install"
    else
        fail "recallforge CLI broken with CUDA install"
    fi

    deactivate
else
    skip "CUDA install (no CUDA GPU)"
fi

# ──────────────────────────────────────────────
print_summary "Install Tests"
