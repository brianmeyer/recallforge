"""Tests for VL-aware query expansion (REC-118)."""

import pytest
from unittest.mock import MagicMock

from recallforge.search import (
    expand_query,
    _is_visual_query,
    _generate_text_variants,
    _generate_visual_description,
    _VISUAL_PHRASE_INDICATORS,
    HybridSearcher,
)


class BackendWithoutGenerateText:
    """Test helper that forces the heuristic expansion path."""

    generate_text = None


class BackendWithGenerateText:
    """Test helper for generated expansion path."""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate_text(self, prompt: str, max_tokens: int = 60) -> str:
        self.calls.append((prompt, max_tokens))
        return self.response


class TestVisualQueryDetection:
    """Test visual query detection."""

    @pytest.mark.parametrize("query", [
        "show me the diagram",
        "image of a neural network",
        "photo of the team",
        "picture of the architecture",
        "screenshot of the error",
        "whiteboard from the meeting",
        "that photo of the event",
        "the diagram in the document",
    ])
    def test_detects_visual_queries(self, query):
        """Visual queries should be detected."""
        assert _is_visual_query(query) is True

    @pytest.mark.parametrize("query", [
        "how do vector embeddings work",
        "what is the best approach",
        "explain transformer architecture",
        "python tutorial for beginners",
        "machine learning concepts",
        "data structures and algorithms",
    ])
    def test_non_visual_queries(self, query):
        """Non-visual queries should not be detected as visual."""
        assert _is_visual_query(query) is False

    def test_visual_indicators_list_not_empty(self):
        """Visual indicators list should be populated."""
        assert len(_VISUAL_PHRASE_INDICATORS) > 0
        assert all(isinstance(ind, str) for ind in _VISUAL_PHRASE_INDICATORS)


class TestTextVariantGeneration:
    """Test text query variant generation."""

    def test_generates_variants_for_how_to(self):
        """Should generate variants for 'how to' queries."""
        backend = BackendWithoutGenerateText()
        variants = _generate_text_variants("how to make pasta", backend)
        assert len(variants) > 0
        assert any("guide" in v or "steps" in v or "tutorial" in v for v in variants)

    def test_generates_variants_for_what_is(self):
        """Should generate variants for 'what is' queries."""
        backend = BackendWithoutGenerateText()
        variants = _generate_text_variants("what is machine learning", backend)
        assert len(variants) > 0
        assert any("definition" in v or "explaining" in v for v in variants)

    def test_generates_variants_for_difference(self):
        """Should generate variants for comparison queries."""
        backend = BackendWithoutGenerateText()
        variants = _generate_text_variants("difference between python and javascript", backend)
        assert len(variants) > 0
        assert any("comparison" in v or "vs" in v for v in variants)

    def test_limits_to_two_variants(self):
        """Should return at most 2 variants."""
        backend = BackendWithoutGenerateText()
        variants = _generate_text_variants("how to best way to example of something", backend)
        assert len(variants) <= 2

    def test_no_variants_for_simple_queries(self):
        """Simple queries without patterns may not generate variants."""
        backend = BackendWithoutGenerateText()
        variants = _generate_text_variants("simple query", backend)
        # May be empty or have rephrased variants
        assert isinstance(variants, list)

    def test_prefers_generated_variants_when_backend_supports_it(self):
        """Generated expansions should be used before heuristic rules."""
        backend = BackendWithGenerateText("pasta cooking guide\npasta recipe steps")

        variants = _generate_text_variants("how to make pasta", backend)

        assert variants == ["pasta cooking guide", "pasta recipe steps"]
        assert len(backend.calls) == 1
        prompt, max_tokens = backend.calls[0]
        assert "Rewrite the following search query" in prompt
        assert max_tokens == 80

    def test_parses_json_generated_variants_and_dedupes_original(self):
        """JSON output from the generator should be parsed and cleaned."""
        backend = BackendWithGenerateText(
            '["simple query", "simple query", "better wording", "alternate phrasing"]'
        )

        variants = _generate_text_variants("simple query", backend)

        assert variants == ["better wording", "alternate phrasing"]

    def test_falls_back_to_heuristics_when_generator_output_is_unusable(self):
        """Empty or unusable generation output should not block fallback behavior."""
        backend = BackendWithGenerateText("Here are two options:\n")

        variants = _generate_text_variants("how to make pasta", backend)

        assert len(variants) > 0
        assert any("guide" in v or "steps" in v or "tutorial" in v for v in variants)


class TestVisualDescriptionGeneration:
    """Test visual description generation."""

    def test_generates_description_for_visual_query(self):
        """Should generate image description for visual queries."""
        desc = _generate_visual_description("show me the neural network diagram")
        assert "photograph" in desc.lower() or "image" in desc.lower() or "visual" in desc.lower()
        assert "neural network" in desc.lower()

    def test_generates_description_for_photo_query(self):
        """Should generate description for photo queries."""
        desc = _generate_visual_description("photo of the team")
        assert "team" in desc.lower()

    def test_returns_original_if_no_core_subject(self):
        """Should return original query if no core subject extracted."""
        desc = _generate_visual_description("show me")
        assert desc == "show me"

    def test_strips_visual_indicators(self):
        """Should strip visual indicators from query."""
        desc = _generate_visual_description("diagram of system architecture")
        assert "diagram" not in desc.lower() or "system architecture" in desc.lower()


class TestExpandQuery:
    """Test the main expand_query function."""

    def test_returns_original_when_expand_disabled(self):
        """When expand=False, should return just the original query."""
        backend = BackendWithoutGenerateText()
        result = expand_query("how to make pasta", backend, expand=False)
        assert result == ["how to make pasta"]

    def test_returns_original_for_empty_query(self):
        """Should handle empty queries gracefully."""
        backend = BackendWithoutGenerateText()
        result = expand_query("", backend, expand=True)
        assert result == []

    def test_returns_original_for_none_query(self):
        """Should handle None queries gracefully."""
        backend = BackendWithoutGenerateText()
        result = expand_query(None, backend, expand=True)  # type: ignore
        assert result == []

    def test_expands_visual_query(self):
        """Visual queries should get visual description variant."""
        backend = BackendWithoutGenerateText()
        result = expand_query("show me the neural network", backend, expand=True)
        assert len(result) >= 1
        assert result[0] == "show me the neural network"
        # Should have at least one variant
        if len(result) > 1:
            assert any("photograph" in r or "image" in r or "visual" in r for r in result[1:])

    def test_expands_text_query(self):
        """Text queries should get semantic variants."""
        backend = BackendWithGenerateText("pasta guide\npasta tutorial")
        result = expand_query("how to make pasta", backend, expand=True)
        assert result == ["how to make pasta", "pasta guide", "pasta tutorial"]

    def test_original_always_first(self):
        """Original query should always be first in results."""
        backend = BackendWithoutGenerateText()
        result = expand_query("some query", backend, expand=True)
        assert result[0] == "some query"


class TestHybridSearcherExpandParameter:
    """Test HybridSearcher with expand parameter."""

    def test_searcher_accepts_expand_parameter(self):
        """HybridSearcher should accept expand parameter."""
        backend = MagicMock()
        storage = MagicMock()

        # Should not raise
        searcher = HybridSearcher(
            backend=backend,
            storage=storage,
            expand=True,
        )
        assert searcher.expand is True

        searcher_false = HybridSearcher(
            backend=backend,
            storage=storage,
            expand=False,
        )
        assert searcher_false.expand is False

    def test_searcher_defaults_expand_to_false(self):
        """HybridSearcher should default expand to False."""
        backend = MagicMock()
        storage = MagicMock()

        searcher = HybridSearcher(
            backend=backend,
            storage=storage,
        )
        assert searcher.expand is False


class TestQueryExpansionIntegration:
    """Integration tests for query expansion in search pipeline."""

    def test_expansion_disabled_by_default(self):
        """Query expansion should be opt-in (disabled by default)."""
        # This test verifies the default behavior
        backend = BackendWithoutGenerateText()

        # Default expand=False should return just original
        result = expand_query("test query", backend, expand=False)
        assert len(result) == 1
        assert result[0] == "test query"

    def test_visual_indicators_comprehensive(self):
        """Visual indicators should cover common visual query patterns."""
        # Phrase indicators (exact substring match)
        phrase_queries = [
            "show me", "image of", "photo of", "picture of",
            "architecture diagram", "flow chart", "mind map",
            "that photo", "that image", "the diagram", "the chart",
            "the picture", "the screenshot",
        ]
        # Word indicators (word-boundary match)
        word_queries = [
            "diagram", "screenshot", "illustration", "drawing",
            "infographic", "wireframe", "mockup", "sketch",
            "whiteboard", "portrait",
        ]

        for query in phrase_queries + word_queries:
            assert _is_visual_query(query), f"Should detect: {query}"

    def test_visual_false_positive_rejection(self):
        """Ambiguous words should NOT trigger visual detection as substrings."""
        non_visual = [
            "mapreduce tuning guide",
            "how to show results in terminal",
            "tableau dashboard setup",
            "graph database indexing",
            "figure out the budget",
            "landscape of AI startups",
            "scene understanding in NLP",
        ]
        for query in non_visual:
            assert not _is_visual_query(query), f"Should NOT detect: {query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
