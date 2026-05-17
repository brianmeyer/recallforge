#!/usr/bin/env python3
"""Cross-modal pipeline ablation benchmark for RecallForge.

Tests REAL cross-modal retrieval using actual images, text docs, and documents
from the UAT corpus. Measures what RecallForge is actually built for:
vision-language retrieval where text queries find images and vice versa.

Stages:
1. Vector-only (embed mode, no reranker)
2. BM25-only (full-text search)
3. Vector + BM25 via RRF
4. Vector + BM25 + Reranker (full hybrid)

Categories:
- text_to_text: text query → text document
- text_to_image: text query → relevant image (CROSS-MODAL)
- image_to_text: image as query → relevant text document (CROSS-MODAL)
- mixed_modal: query that should find BOTH text and image results

Metrics: R@1, R@5, R@10, P@5, P@10, NDCG@10, MRR, latency p50/p95
Difficulty tiers: easy, medium, hard with per-tier reporting
Graded relevance: 0 (irrelevant), 1 (partial), 2 (fully relevant)
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recallforge import __version__

CORPUS_DIR = PROJECT_ROOT / "tests" / "uat" / "corpus"
CORPUS_TEXT = CORPUS_DIR / "text"
CORPUS_IMAGES = CORPUS_DIR / "images"
CORPUS_VIDEOS = CORPUS_DIR / "videos"
CORPUS_DOCUMENTS = CORPUS_DIR / "documents"

# ---------------------------------------------------------------------------
# Corpus document registry - all available documents
# ---------------------------------------------------------------------------

CORPUS_DOCS = {
    # Original text documents
    "text/ai_embeddings.md",
    "text/ai_transformers.md",
    "text/ai_agents.md",
    "text/cooking_pasta.md",
    "text/cooking_sourdough.md",
    "text/cooking_grilling.md",
    "text/nature_forests.md",
    "text/nature_mountains.md",
    "text/nature_oceans.md",
    "text/architecture_gothic.md",
    "text/architecture_blueprints.md",
    "text/architecture_modern.md",
    "text/sports_basketball.md",
    "text/sports_running.md",
    "text/sports_soccer.md",
    # New expanded corpus
    "text/tech_quantum_computing.md",
    "text/tech_cybersecurity.md",
    "text/tech_blockchain.md",
    "text/tech_cloud_computing.md",
    "text/tech_edge_ai.md",
    "text/science_genetics.md",
    "text/science_climate.md",
    "text/science_astronomy.md",
    "text/science_neuroscience.md",
    "text/science_renewable_energy.md",
    "text/cooking_baking.md",
    "text/cooking_asian_cuisine.md",
    "text/cooking_bbq.md",
    "text/cooking_desserts.md",
    "text/cooking_spices.md",
    "text/sports_tennis.md",
    "text/sports_swimming.md",
    "text/sports_cycling.md",
    "text/sports_golf.md",
    "text/sports_yoga.md",
    "text/history_ancient_rome.md",
    "text/history_renaissance.md",
    "text/history_industrial_revolution.md",
    "text/history_ww2.md",
    "text/history_space_race.md",
    "text/medicine_cardiology.md",
    "text/medicine_nutrition.md",
    "text/medicine_mental_health.md",
    "text/medicine_immunology.md",
    "text/medicine_surgery.md",
    "text/music_classical.md",
    "text/music_jazz.md",
    "text/music_production.md",
    "text/art_impressionism.md",
    "text/art_photography.md",
    "text/finance_investing.md",
    "text/finance_cryptocurrency.md",
    "text/travel_japan.md",
    "text/travel_national_parks.md",
    # Images
    "images/neural_network_diagram.png",
    "images/whiteboard_architecture.png",
    "images/whiteboard_brainstorm.png",
    "images/food_pasta_dish.png",
    "images/forest_landscape.png",
    "images/mountain_landscape.png",
    "images/ocean_beach.png",
    "images/floor_plan_blueprint.png",
    "images/code_editor_screenshot.png",
    "images/handwritten_notes.png",
    # Videos
    "videos/architecture_walkthrough.mp4",
    "videos/coding_demo.mp4",
    "videos/cooking_tutorial.mp4",
    "videos/nature_timelapse.mp4",
    "videos/whiteboard_session.mp4",
    # Video transcripts
    "videos/architecture_walkthrough.transcript.json",
    "videos/coding_demo.transcript.json",
    "videos/cooking_tutorial.transcript.json",
    "videos/nature_timelapse.transcript.json",
    "videos/whiteboard_session.transcript.json",
    # Documents (PDF, DOCX, PPTX)
    "documents/ai_strategy_report.docx",
    "documents/project_status_q1.docx",
    "documents/recallforge_spec.docx",
    "documents/ai_architecture_deck.pptx",
    "documents/quarterly_review.pptx",
    "documents/embedding_research.pdf",
    "documents/operations_manual.pdf",
    "documents/edge_deployment_guide.pdf",
}

# ---------------------------------------------------------------------------
# Ground truth definitions — expanded cross-modal pairs with difficulty and graded relevance
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """A query with its expected relevant documents and metadata."""
    query: str
    relevant_paths: List[str]  # paths relative to corpus
    category: str
    query_type: str = "text"  # "text", "image", or "video"
    image_query_path: Optional[str] = None  # for image-as-query
    video_query_path: Optional[str] = None  # for video-as-query
    difficulty: str = "medium"  # "easy", "medium", "hard"
    graded_relevance: Optional[Dict[str, int]] = None  # path -> 0/1/2 relevance score
    is_negative_control: bool = False  # query with zero expected results

    def get_relevance_score(self, path: str) -> int:
        """Get graded relevance score for a path (0=irrelevant, 1=partial, 2=full)."""
        if self.graded_relevance is None:
            return 2 if path in self.relevant_paths else 0
        return self.graded_relevance.get(path, 0)


WEAK_CROSS_MODAL_BENCHMARK_CATEGORIES = {
    "image_to_text",
    "image_to_document",
    "video_to_text",
    "video_to_image",
    "video_to_document",
}

MIN_WEAK_CATEGORY_QUERIES = 20


def _media_query_variants(
    *,
    category: str,
    query_type: str,
    source_path: str,
    relevant_paths: List[str],
    queries: List[str],
    difficulty: str,
    graded_relevance: Optional[Dict[str, int]] = None,
) -> List[GroundTruth]:
    """Build grounded media-query variants without duplicating path metadata."""
    variants: List[GroundTruth] = []
    for query in queries:
        kwargs: Dict[str, Any] = {
            "query": query,
            "query_type": query_type,
            "relevant_paths": relevant_paths,
            "category": category,
            "difficulty": difficulty,
            "graded_relevance": graded_relevance,
        }
        if query_type == "image":
            kwargs["image_query_path"] = source_path
        elif query_type == "video":
            kwargs["video_query_path"] = source_path
        else:
            raise ValueError(f"Unsupported media query_type: {query_type}")
        variants.append(GroundTruth(**kwargs))
    return variants


# ============================================================================
# TEXT → TEXT QUERIES (50 queries)
# ============================================================================

TEXT_TO_TEXT = [
    # EASY (20 queries) - direct keyword matches
    GroundTruth(
        query="how do vector embeddings work for semantic search",
        relevant_paths=["text/ai_embeddings.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="transformer architecture attention mechanism",
        relevant_paths=["text/ai_transformers.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="autonomous AI agents with episodic memory",
        relevant_paths=["text/ai_agents.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="how to make fresh pasta dough at home",
        relevant_paths=["text/cooking_pasta.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="sourdough starter feeding schedule",
        relevant_paths=["text/cooking_sourdough.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="charcoal grill two zone setup searing",
        relevant_paths=["text/cooking_grilling.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="temperate deciduous forest ecosystem oak maple",
        relevant_paths=["text/nature_forests.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="mountain alpine ecology tectonic plates",
        relevant_paths=["text/nature_mountains.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="marine biology coral reef phytoplankton",
        relevant_paths=["text/nature_oceans.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="gothic cathedral flying buttresses ribbed vaults",
        relevant_paths=["text/architecture_gothic.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="reading architectural blueprints floor plans elevations",
        relevant_paths=["text/architecture_blueprints.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="Le Corbusier modern architecture glass curtain walls",
        relevant_paths=["text/architecture_modern.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="NBA basketball three point shooting analytics",
        relevant_paths=["text/sports_basketball.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="marathon training plan progressive mileage tempo runs",
        relevant_paths=["text/sports_running.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="soccer formations 4-3-3 wingbacks tactics",
        relevant_paths=["text/sports_soccer.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="quantum computing qubits superposition",
        relevant_paths=["text/tech_quantum_computing.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="cybersecurity zero trust architecture MFA",
        relevant_paths=["text/tech_cybersecurity.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="blockchain smart contracts Ethereum",
        relevant_paths=["text/tech_blockchain.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="cloud computing AWS Kubernetes microservices",
        relevant_paths=["text/tech_cloud_computing.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    GroundTruth(
        query="edge AI on-device machine learning TensorFlow Lite",
        relevant_paths=["text/tech_edge_ai.md"],
        category="text_to_text",
        difficulty="easy",
    ),
    
    # MEDIUM (20 queries) - requires semantic understanding
    GroundTruth(
        query="how do computers understand the meaning of words",
        relevant_paths=["text/ai_embeddings.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="why do transformers process sequences in parallel",
        relevant_paths=["text/ai_transformers.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="systems that remember past interactions",
        relevant_paths=["text/ai_agents.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="fermented wheat dough with wild yeast culture",
        relevant_paths=["text/cooking_sourdough.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="tall trees with seasonal leaf changes",
        relevant_paths=["text/nature_forests.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="high altitude environments with snow caps",
        relevant_paths=["text/nature_mountains.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="saltwater ecosystems with coral reefs",
        relevant_paths=["text/nature_oceans.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="medieval church building techniques",
        relevant_paths=["text/architecture_gothic.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="how to read construction drawings",
        relevant_paths=["text/architecture_blueprints.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="professional basketball strategy and analytics",
        relevant_paths=["text/sports_basketball.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="long distance running preparation plans",
        relevant_paths=["text/sports_running.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="football team positioning and strategy",
        relevant_paths=["text/sports_soccer.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="DNA sequencing and gene editing CRISPR",
        relevant_paths=["text/science_genetics.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="global warming carbon emissions renewable energy",
        relevant_paths=["text/science_climate.md", "text/science_renewable_energy.md"],
        category="text_to_text",
        difficulty="medium",
        graded_relevance={"text/science_climate.md": 2, "text/science_renewable_energy.md": 1},
    ),
    GroundTruth(
        query="telescopes observing distant galaxies",
        relevant_paths=["text/science_astronomy.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="brain neurons synapses memory formation",
        relevant_paths=["text/science_neuroscience.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="bread making fermentation gluten development",
        relevant_paths=["text/cooking_baking.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="wok cooking stir fry Asian flavors",
        relevant_paths=["text/cooking_asian_cuisine.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="smoking meat low and slow barbecue",
        relevant_paths=["text/cooking_bbq.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="French pastry croissants laminated dough",
        relevant_paths=["text/cooking_desserts.md"],
        category="text_to_text",
        difficulty="medium",
    ),
    
    # HARD (10 queries) - abstract or indirect matches
    GroundTruth(
        query="the alignment problem in machine learning systems",
        relevant_paths=["text/ai_agents.md", "text/ai_transformers.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/ai_agents.md": 2, "text/ai_transformers.md": 1},
    ),
    GroundTruth(
        query="why do some baked goods rise while others fall flat",
        relevant_paths=["text/cooking_baking.md", "text/cooking_sourdough.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/cooking_baking.md": 2, "text/cooking_sourdough.md": 1},
    ),
    GroundTruth(
        query="how civilizations build lasting monuments",
        relevant_paths=["text/history_ancient_rome.md", "text/architecture_gothic.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/history_ancient_rome.md": 2, "text/architecture_gothic.md": 1},
    ),
    GroundTruth(
        query="the intersection of art and science in understanding nature",
        relevant_paths=["text/art_impressionism.md", "text/science_astronomy.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/art_impressionism.md": 2, "text/science_astronomy.md": 1},
    ),
    GroundTruth(
        query="competition between nations driving technological advancement",
        relevant_paths=["text/history_space_race.md", "text/tech_quantum_computing.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/history_space_race.md": 2, "text/tech_quantum_computing.md": 1},
    ),
    GroundTruth(
        query="how the body defends against invaders",
        relevant_paths=["text/medicine_immunology.md"],
        category="text_to_text",
        difficulty="hard",
    ),
    GroundTruth(
        query="the role of rhythm in human expression",
        relevant_paths=["text/music_jazz.md", "text/music_classical.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/music_jazz.md": 2, "text/music_classical.md": 1},
    ),
    GroundTruth(
        query="preserving moments through visual recording",
        relevant_paths=["text/art_photography.md"],
        category="text_to_text",
        difficulty="hard",
    ),
    GroundTruth(
        query="managing risk while seeking returns",
        relevant_paths=["text/finance_investing.md"],
        category="text_to_text",
        difficulty="hard",
    ),
    GroundTruth(
        query="healing through mindfulness and movement",
        relevant_paths=["text/sports_yoga.md", "text/medicine_mental_health.md"],
        category="text_to_text",
        difficulty="hard",
        graded_relevance={"text/sports_yoga.md": 2, "text/medicine_mental_health.md": 1},
    ),
]

# ============================================================================
# TEXT → IMAGE QUERIES (20 queries)
# ============================================================================

TEXT_TO_IMAGE = [
    # EASY (8 queries)
    GroundTruth(
        query="neural network layers diagram with connections",
        relevant_paths=["images/neural_network_diagram.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="system architecture diagram with API gateway and database",
        relevant_paths=["images/whiteboard_architecture.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="AI strategy brainstorming mind map whiteboard",
        relevant_paths=["images/whiteboard_brainstorm.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="plate of pasta with tomato sauce and basil",
        relevant_paths=["images/food_pasta_dish.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="forest with tall trees and walking path",
        relevant_paths=["images/forest_landscape.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="mountain landscape with snow peaks and meadow",
        relevant_paths=["images/mountain_landscape.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="ocean waves beach sand and shells",
        relevant_paths=["images/ocean_beach.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    GroundTruth(
        query="floor plan blueprint with rooms and dimensions",
        relevant_paths=["images/floor_plan_blueprint.png"],
        category="text_to_image",
        difficulty="easy",
    ),
    
    # MEDIUM (8 queries)
    GroundTruth(
        query="software development IDE with Python code",
        relevant_paths=["images/code_editor_screenshot.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="handwritten meeting notes sprint review action items",
        relevant_paths=["images/handwritten_notes.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="diagram showing deep learning model architecture",
        relevant_paths=["images/neural_network_diagram.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="Italian cuisine dish photography",
        relevant_paths=["images/food_pasta_dish.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="architectural technical drawing",
        relevant_paths=["images/floor_plan_blueprint.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="nature scenery with coniferous trees",
        relevant_paths=["images/forest_landscape.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="coastal landscape photography",
        relevant_paths=["images/ocean_beach.png"],
        category="text_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="whiteboard session with diagrams and arrows",
        relevant_paths=["images/whiteboard_brainstorm.png", "images/whiteboard_architecture.png"],
        category="text_to_image",
        difficulty="medium",
        graded_relevance={"images/whiteboard_brainstorm.png": 2, "images/whiteboard_architecture.png": 1},
    ),
    
    # HARD (4 queries) - abstract descriptions
    GroundTruth(
        query="visual representation of artificial intelligence concepts",
        relevant_paths=["images/neural_network_diagram.png", "images/whiteboard_brainstorm.png"],
        category="text_to_image",
        difficulty="hard",
        graded_relevance={"images/neural_network_diagram.png": 2, "images/whiteboard_brainstorm.png": 1},
    ),
    GroundTruth(
        query="documentation of outdoor environments",
        relevant_paths=["images/forest_landscape.png", "images/mountain_landscape.png", "images/ocean_beach.png"],
        category="text_to_image",
        difficulty="hard",
        graded_relevance={"images/forest_landscape.png": 2, "images/mountain_landscape.png": 2, "images/ocean_beach.png": 2},
    ),
]

# ============================================================================
# IMAGE → TEXT QUERIES (20 queries)
# ============================================================================

IMAGE_TO_TEXT = [
    # EASY (6 queries)
    GroundTruth(
        query="",
        relevant_paths=["text/ai_transformers.md", "text/ai_embeddings.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/neural_network_diagram.png",
        difficulty="easy",
        graded_relevance={"text/ai_transformers.md": 2, "text/ai_embeddings.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/cooking_pasta.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/food_pasta_dish.png",
        difficulty="easy",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_forests.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/forest_landscape.png",
        difficulty="easy",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_mountains.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/mountain_landscape.png",
        difficulty="easy",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_oceans.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/ocean_beach.png",
        difficulty="easy",
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/architecture_blueprints.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/floor_plan_blueprint.png",
        difficulty="easy",
    ),
    
    # MEDIUM (6 queries)
    GroundTruth(
        query="",
        relevant_paths=["text/ai_agents.md", "text/tech_edge_ai.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/whiteboard_architecture.png",
        difficulty="medium",
        graded_relevance={"text/ai_agents.md": 2, "text/tech_edge_ai.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/whiteboard_brainstorm.png",
        difficulty="medium",
        graded_relevance={"text/ai_agents.md": 2, "text/tech_cloud_computing.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/tech_edge_ai.md", "text/ai_embeddings.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/code_editor_screenshot.png",
        difficulty="medium",
        graded_relevance={"text/tech_edge_ai.md": 2, "text/ai_embeddings.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/ai_agents.md", "text/sports_running.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/handwritten_notes.png",
        difficulty="medium",
        graded_relevance={"text/ai_agents.md": 2, "text/sports_running.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/cooking_asian_cuisine.md", "text/cooking_spices.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/food_pasta_dish.png",
        difficulty="medium",
        graded_relevance={"text/cooking_asian_cuisine.md": 1, "text/cooking_spices.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/architecture_modern.md", "text/architecture_gothic.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/floor_plan_blueprint.png",
        difficulty="medium",
        graded_relevance={"text/architecture_modern.md": 1, "text/architecture_gothic.md": 1},
    ),
    
    # HARD (3 queries)
    GroundTruth(
        query="",
        relevant_paths=["text/ai_transformers.md", "text/tech_edge_ai.md", "text/ai_agents.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/neural_network_diagram.png",
        difficulty="hard",
        graded_relevance={"text/ai_transformers.md": 2, "text/tech_edge_ai.md": 1, "text/ai_agents.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/nature_forests.md", "text/nature_mountains.md", "text/nature_oceans.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/mountain_landscape.png",
        difficulty="hard",
        graded_relevance={"text/nature_mountains.md": 2, "text/nature_forests.md": 1, "text/nature_oceans.md": 1},
    ),
    GroundTruth(
        query="",
        relevant_paths=["text/tech_cybersecurity.md", "text/tech_blockchain.md", "text/tech_cloud_computing.md"],
        category="image_to_text",
        query_type="image",
        image_query_path="images/whiteboard_architecture.png",
        difficulty="hard",
        graded_relevance={"text/tech_cybersecurity.md": 1, "text/tech_blockchain.md": 1, "text/tech_cloud_computing.md": 1},
    ),
]

IMAGE_TO_TEXT += _media_query_variants(
    category="image_to_text",
    query_type="image",
    source_path="images/neural_network_diagram.png",
    relevant_paths=["text/ai_transformers.md", "text/ai_embeddings.md", "text/science_neuroscience.md"],
    queries=["neural network diagram connected to transformer and embedding notes"],
    difficulty="medium",
    graded_relevance={"text/ai_transformers.md": 2, "text/ai_embeddings.md": 2, "text/science_neuroscience.md": 1},
)
IMAGE_TO_TEXT += _media_query_variants(
    category="image_to_text",
    query_type="image",
    source_path="images/code_editor_screenshot.png",
    relevant_paths=["text/tech_edge_ai.md", "text/tech_cybersecurity.md", "text/tech_cloud_computing.md"],
    queries=["code editor screenshot linked to edge AI and cloud engineering notes"],
    difficulty="medium",
    graded_relevance={"text/tech_edge_ai.md": 2, "text/tech_cloud_computing.md": 1, "text/tech_cybersecurity.md": 1},
)
IMAGE_TO_TEXT += _media_query_variants(
    category="image_to_text",
    query_type="image",
    source_path="images/handwritten_notes.png",
    relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md", "text/finance_investing.md"],
    queries=["handwritten planning notes connected to agent roadmap and budget context"],
    difficulty="hard",
    graded_relevance={"text/ai_agents.md": 2, "text/tech_cloud_computing.md": 1, "text/finance_investing.md": 1},
)
IMAGE_TO_TEXT += _media_query_variants(
    category="image_to_text",
    query_type="image",
    source_path="images/whiteboard_brainstorm.png",
    relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md", "text/tech_edge_ai.md"],
    queries=["brainstorm whiteboard should retrieve agent architecture notes"],
    difficulty="hard",
    graded_relevance={"text/ai_agents.md": 2, "text/tech_cloud_computing.md": 1, "text/tech_edge_ai.md": 1},
)
IMAGE_TO_TEXT += _media_query_variants(
    category="image_to_text",
    query_type="image",
    source_path="images/floor_plan_blueprint.png",
    relevant_paths=["text/architecture_blueprints.md", "text/architecture_modern.md", "text/architecture_gothic.md"],
    queries=["floor plan image tied to architecture blueprint reference notes"],
    difficulty="medium",
    graded_relevance={"text/architecture_blueprints.md": 2, "text/architecture_modern.md": 1, "text/architecture_gothic.md": 1},
)

# ============================================================================
# MIXED MODAL QUERIES (20 queries) - find BOTH text and image
# ============================================================================

MIXED_MODAL = [
    # EASY (8 queries)
    GroundTruth(
        query="pasta cooking recipe and food photo",
        relevant_paths=["text/cooking_pasta.md", "images/food_pasta_dish.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="forest ecology trees landscape",
        relevant_paths=["text/nature_forests.md", "images/forest_landscape.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="mountain environment alpine peaks meadow",
        relevant_paths=["text/nature_mountains.md", "images/mountain_landscape.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="ocean marine beach waves",
        relevant_paths=["text/nature_oceans.md", "images/ocean_beach.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="neural network deep learning architecture diagram",
        relevant_paths=["text/ai_transformers.md", "images/neural_network_diagram.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="floor plan building architecture blueprint design",
        relevant_paths=["text/architecture_blueprints.md", "images/floor_plan_blueprint.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="software architecture API gateway backend system design",
        relevant_paths=["text/ai_agents.md", "images/whiteboard_architecture.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    GroundTruth(
        query="AI brainstorming session whiteboard notes",
        relevant_paths=["text/ai_agents.md", "images/whiteboard_brainstorm.png"],
        category="mixed_modal",
        difficulty="easy",
    ),
    
    # MEDIUM (8 queries)
    GroundTruth(
        query="baking bread fermentation process and techniques",
        relevant_paths=["text/cooking_baking.md", "text/cooking_sourdough.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/cooking_baking.md": 2, "text/cooking_sourdough.md": 2},
    ),
    GroundTruth(
        query="barbecue smoking meat techniques wood selection",
        relevant_paths=["text/cooking_bbq.md", "text/cooking_grilling.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/cooking_bbq.md": 2, "text/cooking_grilling.md": 1},
    ),
    GroundTruth(
        query="quantum computing and edge AI hardware",
        relevant_paths=["text/tech_quantum_computing.md", "text/tech_edge_ai.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/tech_quantum_computing.md": 2, "text/tech_edge_ai.md": 2},
    ),
    GroundTruth(
        query="cloud security and blockchain technology",
        relevant_paths=["text/tech_cybersecurity.md", "text/tech_blockchain.md", "text/tech_cloud_computing.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/tech_cybersecurity.md": 2, "text/tech_blockchain.md": 1, "text/tech_cloud_computing.md": 1},
    ),
    GroundTruth(
        query="DNA genetics and neuroscience brain research",
        relevant_paths=["text/science_genetics.md", "text/science_neuroscience.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/science_genetics.md": 2, "text/science_neuroscience.md": 2},
    ),
    GroundTruth(
        query="renewable energy and climate change solutions",
        relevant_paths=["text/science_renewable_energy.md", "text/science_climate.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/science_renewable_energy.md": 2, "text/science_climate.md": 2},
    ),
    GroundTruth(
        query="tennis and golf sports techniques",
        relevant_paths=["text/sports_tennis.md", "text/sports_golf.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/sports_tennis.md": 2, "text/sports_golf.md": 2},
    ),
    GroundTruth(
        query="swimming and cycling endurance training",
        relevant_paths=["text/sports_swimming.md", "text/sports_cycling.md"],
        category="mixed_modal",
        difficulty="medium",
        graded_relevance={"text/sports_swimming.md": 2, "text/sports_cycling.md": 2},
    ),
    
    # HARD (4 queries)
    GroundTruth(
        query="visual and textual documentation of natural environments",
        relevant_paths=["text/nature_forests.md", "text/nature_mountains.md", "text/nature_oceans.md",
                        "images/forest_landscape.png", "images/mountain_landscape.png", "images/ocean_beach.png"],
        category="mixed_modal",
        difficulty="hard",
        graded_relevance={"text/nature_forests.md": 2, "text/nature_mountains.md": 2, "text/nature_oceans.md": 2,
                         "images/forest_landscape.png": 2, "images/mountain_landscape.png": 2, "images/ocean_beach.png": 2},
    ),
    GroundTruth(
        query="artistic and technical representations of architecture",
        relevant_paths=["text/architecture_gothic.md", "text/architecture_modern.md", "text/architecture_blueprints.md",
                        "images/floor_plan_blueprint.png"],
        category="mixed_modal",
        difficulty="hard",
        graded_relevance={"text/architecture_gothic.md": 2, "text/architecture_modern.md": 2, 
                         "text/architecture_blueprints.md": 2, "images/floor_plan_blueprint.png": 2},
    ),
    GroundTruth(
        query="AI system design from concept to implementation",
        relevant_paths=["text/ai_agents.md", "text/ai_transformers.md", "text/tech_edge_ai.md",
                        "images/neural_network_diagram.png", "images/whiteboard_architecture.png", "images/code_editor_screenshot.png"],
        category="mixed_modal",
        difficulty="hard",
        graded_relevance={"text/ai_agents.md": 2, "text/ai_transformers.md": 1, "text/tech_edge_ai.md": 1,
                         "images/neural_network_diagram.png": 2, "images/whiteboard_architecture.png": 2, "images/code_editor_screenshot.png": 1},
    ),
    GroundTruth(
        query="comprehensive guide to athletic performance",
        relevant_paths=["text/sports_running.md", "text/sports_cycling.md", "text/sports_swimming.md",
                        "text/sports_yoga.md", "text/medicine_nutrition.md", "text/medicine_cardiology.md"],
        category="mixed_modal",
        difficulty="hard",
        graded_relevance={"text/sports_running.md": 2, "text/sports_cycling.md": 2, "text/sports_swimming.md": 2,
                         "text/sports_yoga.md": 1, "text/medicine_nutrition.md": 1, "text/medicine_cardiology.md": 1},
    ),
]

# ============================================================================
# TEXT → VIDEO QUERIES (15 queries)
# ============================================================================

TEXT_TO_VIDEO = [
    # EASY (6 queries)
    GroundTruth(
        query="office walkthrough connecting floor plan and system architecture",
        relevant_paths=["videos/architecture_walkthrough.mp4"],
        category="text_to_video",
        difficulty="easy",
    ),
    GroundTruth(
        query="screen recording debugging RecallForge video search test",
        relevant_paths=["videos/coding_demo.mp4"],
        category="text_to_video",
        difficulty="easy",
    ),
    GroundTruth(
        query="family dinner pasta recipe video with handwritten substitutions",
        relevant_paths=["videos/cooking_tutorial.mp4"],
        category="text_to_video",
        difficulty="easy",
    ),
    GroundTruth(
        query="weekend trail scouting video forest mountain coast",
        relevant_paths=["videos/nature_timelapse.mp4"],
        category="text_to_video",
        difficulty="easy",
    ),
    GroundTruth(
        query="product planning whiteboard meeting memory rollups",
        relevant_paths=["videos/whiteboard_session.mp4"],
        category="text_to_video",
        difficulty="easy",
    ),
    GroundTruth(
        query="walkthrough transcript about floor plan architecture deck",
        relevant_paths=["videos/architecture_walkthrough.mp4", "videos/architecture_walkthrough.transcript.json"],
        category="text_to_video",
        difficulty="easy",
        graded_relevance={"videos/architecture_walkthrough.mp4": 2, "videos/architecture_walkthrough.transcript.json": 2},
    ),
    
    # MEDIUM (6 queries)
    GroundTruth(
        query="developer screen recording and meeting notes about search pipeline",
        relevant_paths=["videos/coding_demo.mp4", "videos/whiteboard_session.mp4"],
        category="text_to_video",
        difficulty="medium",
        graded_relevance={"videos/coding_demo.mp4": 2, "videos/whiteboard_session.mp4": 1},
    ),
    GroundTruth(
        query="recipe memory with pasta sauce timing and grocery planning",
        relevant_paths=["videos/cooking_tutorial.mp4"],
        category="text_to_video",
        difficulty="medium",
    ),
    GroundTruth(
        query="outdoor field clip with route planning and park notes",
        relevant_paths=["videos/nature_timelapse.mp4"],
        category="text_to_video",
        difficulty="medium",
    ),
    GroundTruth(
        query="meeting recordings with transcript action items for review",
        relevant_paths=["videos/whiteboard_session.mp4", "videos/whiteboard_session.transcript.json"],
        category="text_to_video",
        difficulty="medium",
        graded_relevance={"videos/whiteboard_session.mp4": 2, "videos/whiteboard_session.transcript.json": 2},
    ),
    GroundTruth(
        query="searchable transcript memories from kitchen and developer videos",
        relevant_paths=["videos/cooking_tutorial.mp4", "videos/cooking_tutorial.transcript.json",
                        "videos/coding_demo.mp4", "videos/coding_demo.transcript.json"],
        category="text_to_video",
        difficulty="medium",
        graded_relevance={"videos/cooking_tutorial.mp4": 2, "videos/cooking_tutorial.transcript.json": 2,
                         "videos/coding_demo.mp4": 2, "videos/coding_demo.transcript.json": 2},
    ),
    GroundTruth(
        query="visual documentation of outdoor spaces and walkthrough locations",
        relevant_paths=["videos/nature_timelapse.mp4", "videos/architecture_walkthrough.mp4"],
        category="text_to_video",
        difficulty="medium",
        graded_relevance={"videos/nature_timelapse.mp4": 2, "videos/architecture_walkthrough.mp4": 1},
    ),
    
    # HARD (3 queries)
    GroundTruth(
        query="episodic videos with procedural learning and follow-up actions",
        relevant_paths=["videos/cooking_tutorial.mp4", "videos/coding_demo.mp4", "videos/whiteboard_session.mp4",
                        "videos/cooking_tutorial.transcript.json", "videos/coding_demo.transcript.json", "videos/whiteboard_session.transcript.json"],
        category="text_to_video",
        difficulty="hard",
        graded_relevance={"videos/cooking_tutorial.mp4": 2, "videos/coding_demo.mp4": 2, "videos/whiteboard_session.mp4": 2,
                         "videos/cooking_tutorial.transcript.json": 2, "videos/coding_demo.transcript.json": 2, "videos/whiteboard_session.transcript.json": 2},
    ),
    GroundTruth(
        query="archived recordings with field notes and architecture narration",
        relevant_paths=["videos/architecture_walkthrough.mp4", "videos/architecture_walkthrough.transcript.json",
                        "videos/nature_timelapse.mp4", "videos/nature_timelapse.transcript.json"],
        category="text_to_video",
        difficulty="hard",
        graded_relevance={"videos/architecture_walkthrough.mp4": 2, "videos/architecture_walkthrough.transcript.json": 2,
                         "videos/nature_timelapse.mp4": 2, "videos/nature_timelapse.transcript.json": 2},
    ),
    GroundTruth(
        query="comprehensive episodic video library with transcripts",
        relevant_paths=["videos/cooking_tutorial.mp4", "videos/coding_demo.mp4", "videos/whiteboard_session.mp4", "videos/architecture_walkthrough.mp4", "videos/nature_timelapse.mp4"],
        category="text_to_video",
        difficulty="hard",
        graded_relevance={"videos/cooking_tutorial.mp4": 2, "videos/coding_demo.mp4": 2, "videos/whiteboard_session.mp4": 2,
                         "videos/architecture_walkthrough.mp4": 2, "videos/nature_timelapse.mp4": 2},
    ),
]

# ============================================================================
# HARD NEGATIVES (queries similar to but NOT about the target)
# ============================================================================

HARD_NEGATIVES = [
    # Similar to pasta but about Asian noodles
    GroundTruth(
        query="ramen noodle soup recipe Japanese cuisine",
        relevant_paths=["text/cooking_asian_cuisine.md"],  # NOT pasta
        category="text_to_text",
        difficulty="medium",
    ),
    # Similar to neural networks but about brain biology
    GroundTruth(
        query="biological neurons in the human brain synapses",
        relevant_paths=["text/science_neuroscience.md"],  # NOT AI
        category="text_to_text",
        difficulty="medium",
    ),
    # Similar to architecture but about software
    GroundTruth(
        query="software design patterns and system architecture",
        relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md"],  # NOT building architecture
        category="text_to_text",
        difficulty="medium",
    ),
    # Similar to blockchain but about traditional banking
    GroundTruth(
        query="investment banking and stock market trading",
        relevant_paths=["text/finance_investing.md"],  # NOT crypto
        category="text_to_text",
        difficulty="medium",
    ),
    # Similar to yoga but about competitive sports
    GroundTruth(
        query="competitive swimming techniques and training",
        relevant_paths=["text/sports_swimming.md"],  # NOT yoga
        category="text_to_text",
        difficulty="easy",
    ),
]

# ============================================================================
# NEGATIVE CONTROLS (queries with zero expected results)
# ============================================================================

NEGATIVE_CONTROLS = [
    GroundTruth(
        query="underwater basket weaving techniques",
        relevant_paths=[],
        category="text_to_text",
        difficulty="easy",
        is_negative_control=True,
    ),
    GroundTruth(
        query="medieval jousting tournament rules and equipment",
        relevant_paths=[],
        category="text_to_text",
        difficulty="easy",
        is_negative_control=True,
    ),
    GroundTruth(
        query="antique clock repair and horology",
        relevant_paths=[],
        category="text_to_text",
        difficulty="easy",
        is_negative_control=True,
    ),
    GroundTruth(
        query="mycology and wild mushroom foraging",
        relevant_paths=[],
        category="text_to_text",
        difficulty="easy",
        is_negative_control=True,
    ),
    GroundTruth(
        query="vintage typewriter maintenance and repair",
        relevant_paths=[],
        category="text_to_text",
        difficulty="easy",
        is_negative_control=True,
    ),
]

# ── Text → Document queries ──────────────────────────────────
TEXT_TO_DOCUMENT = [
    GroundTruth(
        query="enterprise AI strategy roadmap deployment",
        relevant_paths=["documents/ai_strategy_report.docx"],
        category="text_to_document",
        difficulty="easy",
    ),
    GroundTruth(
        query="retrieval augmented generation RAG productivity",
        relevant_paths=["documents/ai_strategy_report.docx"],
        category="text_to_document",
        difficulty="medium",
    ),
    GroundTruth(
        query="defense AI studio quarterly milestones sensor fusion",
        relevant_paths=["documents/project_status_q1.docx"],
        category="text_to_document",
        difficulty="easy",
    ),
    GroundTruth(
        query="$38M portfolio budget burn rate GPU supply chain",
        relevant_paths=["documents/project_status_q1.docx"],
        category="text_to_document",
        difficulty="medium",
    ),
    GroundTruth(
        query="RecallForge multimodal memory search engine technical specification",
        relevant_paths=["documents/recallforge_spec.docx"],
        category="text_to_document",
        difficulty="easy",
    ),
    GroundTruth(
        query="Qwen3-VL embedding reranker captioner model stack",
        relevant_paths=["documents/recallforge_spec.docx", "documents/ai_architecture_deck.pptx"],
        category="text_to_document",
        difficulty="medium",
    ),
    GroundTruth(
        query="multimodal AI architecture presentation slides BM25 vector fusion",
        relevant_paths=["documents/ai_architecture_deck.pptx"],
        category="text_to_document",
        difficulty="easy",
    ),
    GroundTruth(
        query="defense portfolio business review customer satisfaction",
        relevant_paths=["documents/quarterly_review.pptx"],
        category="text_to_document",
        difficulty="medium",
    ),
    GroundTruth(
        query="cross-modal embedding unified memory search research paper abstract",
        relevant_paths=["documents/embedding_research.pdf"],
        category="text_to_document",
        difficulty="easy",
    ),
    GroundTruth(
        query="pip install recallforge configuration environment variables",
        relevant_paths=["documents/operations_manual.pdf"],
        category="text_to_document",
        difficulty="easy",
    ),
    GroundTruth(
        query="edge AI deployment NVIDIA Jetson quantization pruning",
        relevant_paths=["documents/edge_deployment_guide.pdf"],
        category="text_to_document",
        difficulty="medium",
    ),
    GroundTruth(
        query="Raspberry Pi Qualcomm Snapdragon Hexagon hardware acceleration",
        relevant_paths=["documents/edge_deployment_guide.pdf"],
        category="text_to_document",
        difficulty="hard",
    ),
]

# Combine all queries
# ── Image → Image queries (visual similarity) ────────────────
IMAGE_TO_IMAGE = [
    GroundTruth(
        query="similar landscape",
        query_type="image",
        image_query_path="images/forest_landscape.png",
        relevant_paths=["images/mountain_landscape.png", "images/ocean_beach.png"],
        category="image_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="similar whiteboard",
        query_type="image",
        image_query_path="images/whiteboard_architecture.png",
        relevant_paths=["images/whiteboard_brainstorm.png"],
        category="image_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="similar food",
        query_type="image",
        image_query_path="images/food_pasta_dish.png",
        relevant_paths=["images/food_pasta_dish.png"],
        category="image_to_image",
        difficulty="easy",
        is_negative_control=True,  # exact match test
    ),
]

# ── Image → Video queries ─────────────────────────────────────
IMAGE_TO_VIDEO = [
    GroundTruth(
        query="related video",
        query_type="image",
        image_query_path="images/forest_landscape.png",
        relevant_paths=["videos/nature_timelapse.mp4"],
        category="image_to_video",
        difficulty="hard",
    ),
    GroundTruth(
        query="related video",
        query_type="image",
        image_query_path="images/whiteboard_architecture.png",
        relevant_paths=["videos/architecture_walkthrough.mp4", "videos/whiteboard_session.mp4"],
        category="image_to_video",
        difficulty="hard",
    ),
]

# ── Image → Document queries ──────────────────────────────────
IMAGE_TO_DOCUMENT = [
    GroundTruth(
        query="related document",
        query_type="image",
        image_query_path="images/neural_network_diagram.png",
        relevant_paths=["documents/ai_strategy_report.docx", "documents/ai_architecture_deck.pptx", "documents/embedding_research.pdf"],
        category="image_to_document",
        difficulty="hard",
    ),
    GroundTruth(
        query="related document",
        query_type="image",
        image_query_path="images/code_editor_screenshot.png",
        relevant_paths=["documents/recallforge_spec.docx", "documents/operations_manual.pdf"],
        category="image_to_document",
        difficulty="hard",
    ),
]

IMAGE_TO_DOCUMENT += _media_query_variants(
    category="image_to_document",
    query_type="image",
    source_path="images/neural_network_diagram.png",
    relevant_paths=["documents/ai_strategy_report.docx", "documents/ai_architecture_deck.pptx", "documents/embedding_research.pdf"],
    queries=[
        "neural network diagram should recall AI strategy document",
        "embedding architecture image linked to research paper and slides",
        "visual AI model graph connected to cross-modal embedding documents",
    ],
    difficulty="hard",
    graded_relevance={"documents/ai_strategy_report.docx": 2, "documents/ai_architecture_deck.pptx": 2, "documents/embedding_research.pdf": 2},
)
IMAGE_TO_DOCUMENT += _media_query_variants(
    category="image_to_document",
    query_type="image",
    source_path="images/whiteboard_architecture.png",
    relevant_paths=["documents/recallforge_spec.docx", "documents/ai_architecture_deck.pptx", "documents/ai_strategy_report.docx"],
    queries=[
        "whiteboard architecture sketch should find RecallForge specification",
        "system design drawing linked to architecture presentation",
        "AI service diagram should retrieve strategy roadmap docs",
    ],
    difficulty="hard",
    graded_relevance={"documents/recallforge_spec.docx": 2, "documents/ai_architecture_deck.pptx": 2, "documents/ai_strategy_report.docx": 1},
)
IMAGE_TO_DOCUMENT += _media_query_variants(
    category="image_to_document",
    query_type="image",
    source_path="images/whiteboard_brainstorm.png",
    relevant_paths=["documents/ai_strategy_report.docx", "documents/project_status_q1.docx", "documents/quarterly_review.pptx"],
    queries=[
        "brainstorm board linked to AI roadmap and quarterly review",
        "planning whiteboard should recall project status documentation",
        "strategy workshop image connected to portfolio review slides",
    ],
    difficulty="hard",
    graded_relevance={"documents/ai_strategy_report.docx": 2, "documents/project_status_q1.docx": 2, "documents/quarterly_review.pptx": 1},
)
IMAGE_TO_DOCUMENT += _media_query_variants(
    category="image_to_document",
    query_type="image",
    source_path="images/code_editor_screenshot.png",
    relevant_paths=["documents/recallforge_spec.docx", "documents/operations_manual.pdf", "documents/edge_deployment_guide.pdf"],
    queries=[
        "developer screenshot should retrieve RecallForge technical specification",
        "code editor image linked to install and operations manual",
        "local deployment screen should find edge deployment guide",
    ],
    difficulty="hard",
    graded_relevance={"documents/recallforge_spec.docx": 2, "documents/operations_manual.pdf": 2, "documents/edge_deployment_guide.pdf": 1},
)
IMAGE_TO_DOCUMENT += _media_query_variants(
    category="image_to_document",
    query_type="image",
    source_path="images/floor_plan_blueprint.png",
    relevant_paths=["documents/ai_architecture_deck.pptx", "documents/operations_manual.pdf", "documents/project_status_q1.docx"],
    queries=[
        "blueprint image connected to architecture presentation",
        "floor plan drawing should recall operations and planning docs",
        "building layout visual tied to project review documentation",
    ],
    difficulty="hard",
    graded_relevance={"documents/ai_architecture_deck.pptx": 2, "documents/operations_manual.pdf": 1, "documents/project_status_q1.docx": 1},
)
IMAGE_TO_DOCUMENT += _media_query_variants(
    category="image_to_document",
    query_type="image",
    source_path="images/handwritten_notes.png",
    relevant_paths=["documents/project_status_q1.docx", "documents/quarterly_review.pptx", "documents/ai_strategy_report.docx"],
    queries=[
        "handwritten meeting notes should find project status report",
        "notebook planning image linked to quarterly review deck",
        "planning notes connected to AI strategy report",
    ],
    difficulty="hard",
    graded_relevance={"documents/project_status_q1.docx": 2, "documents/quarterly_review.pptx": 2, "documents/ai_strategy_report.docx": 1},
)

# ── Video → Text queries ──────────────────────────────────────
VIDEO_TO_TEXT = [
    GroundTruth(
        query="related text",
        query_type="video",
        video_query_path="videos/nature_timelapse.mp4",
        relevant_paths=["text/nature_forests.md", "text/nature_mountains.md", "text/nature_oceans.md"],
        category="video_to_text",
        difficulty="medium",
    ),
    GroundTruth(
        query="related text",
        query_type="video",
        video_query_path="videos/coding_demo.mp4",
        relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md", "text/ai_embeddings.md"],
        category="video_to_text",
        difficulty="hard",
    ),
    GroundTruth(
        query="related text",
        query_type="video",
        video_query_path="videos/architecture_walkthrough.mp4",
        relevant_paths=["text/architecture_gothic.md", "text/architecture_modern.md", "text/architecture_blueprints.md"],
        category="video_to_text",
        difficulty="medium",
    ),
]

VIDEO_TO_TEXT += _media_query_variants(
    category="video_to_text",
    query_type="video",
    source_path="videos/nature_timelapse.mp4",
    relevant_paths=["text/nature_forests.md", "text/nature_mountains.md", "text/nature_oceans.md", "text/science_climate.md", "text/travel_national_parks.md"],
    queries=[
        "nature timelapse should recall forest mountain and ocean notes",
        "outdoor landscape video connected to climate and park memories",
        "scenic video with forests mountains and water reference text",
        "environment footage linked to nature and national parks docs",
    ],
    difficulty="medium",
    graded_relevance={"text/nature_forests.md": 2, "text/nature_mountains.md": 2, "text/nature_oceans.md": 2, "text/science_climate.md": 1, "text/travel_national_parks.md": 1},
)
VIDEO_TO_TEXT += _media_query_variants(
    category="video_to_text",
    query_type="video",
    source_path="videos/coding_demo.mp4",
    relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md", "text/tech_edge_ai.md", "text/tech_cybersecurity.md"],
    queries=[
        "coding demo should retrieve software agents and cloud notes",
        "developer video connected to backend architecture text memories",
        "programming walkthrough linked to edge AI implementation notes",
        "software tutorial video should recall cybersecurity and cloud docs",
    ],
    difficulty="hard",
    graded_relevance={"text/ai_agents.md": 2, "text/tech_cloud_computing.md": 2, "text/tech_edge_ai.md": 1, "text/tech_cybersecurity.md": 1},
)
VIDEO_TO_TEXT += _media_query_variants(
    category="video_to_text",
    query_type="video",
    source_path="videos/architecture_walkthrough.mp4",
    relevant_paths=["text/architecture_blueprints.md", "text/architecture_modern.md", "text/architecture_gothic.md", "text/tech_cloud_computing.md"],
    queries=[
        "building walkthrough video connected to architecture reference notes",
        "architecture tour should recall blueprint and modern design text",
        "walkthrough recording linked to systems architecture notes",
    ],
    difficulty="medium",
    graded_relevance={"text/architecture_blueprints.md": 2, "text/architecture_modern.md": 2, "text/architecture_gothic.md": 1, "text/tech_cloud_computing.md": 1},
)
VIDEO_TO_TEXT += _media_query_variants(
    category="video_to_text",
    query_type="video",
    source_path="videos/cooking_tutorial.mp4",
    relevant_paths=["text/cooking_pasta.md", "text/cooking_grilling.md", "text/cooking_spices.md", "text/cooking_asian_cuisine.md"],
    queries=[
        "cooking tutorial video should find recipe and spice notes",
        "food preparation recording connected to pasta cooking memory",
        "culinary demonstration linked to cooking technique documents",
    ],
    difficulty="medium",
    graded_relevance={"text/cooking_pasta.md": 2, "text/cooking_spices.md": 1, "text/cooking_asian_cuisine.md": 1, "text/cooking_grilling.md": 1},
)
VIDEO_TO_TEXT += _media_query_variants(
    category="video_to_text",
    query_type="video",
    source_path="videos/whiteboard_session.mp4",
    relevant_paths=["text/ai_agents.md", "text/tech_cloud_computing.md", "text/tech_edge_ai.md", "text/ai_embeddings.md"],
    queries=[
        "whiteboard session video should retrieve AI agent design notes",
        "planning meeting recording linked to cloud architecture memory",
        "brainstorming video connected to embedding and edge AI notes",
    ],
    difficulty="hard",
    graded_relevance={"text/ai_agents.md": 2, "text/tech_cloud_computing.md": 2, "text/tech_edge_ai.md": 1, "text/ai_embeddings.md": 1},
)

# ── Video → Image queries ─────────────────────────────────────
VIDEO_TO_IMAGE = [
    GroundTruth(
        query="related image",
        query_type="video",
        video_query_path="videos/nature_timelapse.mp4",
        relevant_paths=["images/forest_landscape.png", "images/mountain_landscape.png", "images/ocean_beach.png"],
        category="video_to_image",
        difficulty="medium",
    ),
    GroundTruth(
        query="related image",
        query_type="video",
        video_query_path="videos/whiteboard_session.mp4",
        relevant_paths=["images/whiteboard_brainstorm.png", "images/whiteboard_architecture.png"],
        category="video_to_image",
        difficulty="hard",
    ),
]

VIDEO_TO_IMAGE += _media_query_variants(
    category="video_to_image",
    query_type="video",
    source_path="videos/nature_timelapse.mp4",
    relevant_paths=["images/forest_landscape.png", "images/mountain_landscape.png", "images/ocean_beach.png"],
    queries=[
        "nature video should match forest and mountain images",
        "timelapse landscape recording connected to outdoor photos",
        "scenic footage should recall ocean beach and mountain visuals",
        "environment video linked to landscape image memories",
    ],
    difficulty="medium",
    graded_relevance={"images/forest_landscape.png": 2, "images/mountain_landscape.png": 2, "images/ocean_beach.png": 2},
)
VIDEO_TO_IMAGE += _media_query_variants(
    category="video_to_image",
    query_type="video",
    source_path="videos/whiteboard_session.mp4",
    relevant_paths=["images/whiteboard_brainstorm.png", "images/whiteboard_architecture.png", "images/handwritten_notes.png"],
    queries=[
        "whiteboard meeting video should match brainstorm images",
        "planning session recording connected to whiteboard photos",
        "team board video should recall architecture sketch image",
        "meeting notes video linked to handwritten planning image",
    ],
    difficulty="hard",
    graded_relevance={"images/whiteboard_brainstorm.png": 2, "images/whiteboard_architecture.png": 2, "images/handwritten_notes.png": 1},
)
VIDEO_TO_IMAGE += _media_query_variants(
    category="video_to_image",
    query_type="video",
    source_path="videos/architecture_walkthrough.mp4",
    relevant_paths=["images/floor_plan_blueprint.png", "images/whiteboard_architecture.png", "images/neural_network_diagram.png"],
    queries=[
        "architecture walkthrough should match blueprint and system diagrams",
        "building tour video connected to floor plan visual memory",
        "architecture recording linked to whiteboard design image",
    ],
    difficulty="hard",
    graded_relevance={"images/floor_plan_blueprint.png": 2, "images/whiteboard_architecture.png": 1, "images/neural_network_diagram.png": 1},
)
VIDEO_TO_IMAGE += _media_query_variants(
    category="video_to_image",
    query_type="video",
    source_path="videos/coding_demo.mp4",
    relevant_paths=["images/code_editor_screenshot.png", "images/neural_network_diagram.png", "images/whiteboard_architecture.png"],
    queries=[
        "coding demo should match code editor screenshot",
        "software video connected to neural network diagram image",
        "developer recording linked to architecture whiteboard visual",
    ],
    difficulty="hard",
    graded_relevance={"images/code_editor_screenshot.png": 2, "images/neural_network_diagram.png": 1, "images/whiteboard_architecture.png": 1},
)
VIDEO_TO_IMAGE += _media_query_variants(
    category="video_to_image",
    query_type="video",
    source_path="videos/cooking_tutorial.mp4",
    relevant_paths=["images/food_pasta_dish.png"],
    queries=[
        "cooking tutorial should match pasta dish image",
        "recipe video connected to plated food photo",
        "culinary demonstration linked to food image memory",
        "kitchen video should recall pasta visual",
    ],
    difficulty="medium",
)

# ── Video → Video queries (similarity) ────────────────────────
VIDEO_TO_VIDEO = [
    GroundTruth(
        query="similar video",
        query_type="video",
        video_query_path="videos/nature_timelapse.mp4",
        relevant_paths=["videos/nature_timelapse.mp4"],
        category="video_to_video",
        difficulty="easy",
        is_negative_control=True,  # exact match test
    ),
]

# ── Video → Document queries ──────────────────────────────────
VIDEO_TO_DOCUMENT = [
    GroundTruth(
        query="related document",
        query_type="video",
        video_query_path="videos/architecture_walkthrough.mp4",
        relevant_paths=["documents/ai_architecture_deck.pptx"],
        category="video_to_document",
        difficulty="hard",
    ),
]

VIDEO_TO_DOCUMENT += _media_query_variants(
    category="video_to_document",
    query_type="video",
    source_path="videos/architecture_walkthrough.mp4",
    relevant_paths=["documents/ai_architecture_deck.pptx", "documents/project_status_q1.docx", "documents/ai_strategy_report.docx"],
    queries=[
        "architecture walkthrough should retrieve architecture deck",
        "building tour video connected to planning and status docs",
        "walkthrough recording linked to AI strategy architecture slides",
        "architecture video should recall project milestone deck",
        "tour footage connected to enterprise AI roadmap documents",
        "system walkthrough recording linked to architecture presentation",
        "architecture recording tied to portfolio planning documents",
    ],
    difficulty="hard",
    graded_relevance={"documents/ai_architecture_deck.pptx": 2, "documents/project_status_q1.docx": 1, "documents/ai_strategy_report.docx": 1},
)
VIDEO_TO_DOCUMENT += _media_query_variants(
    category="video_to_document",
    query_type="video",
    source_path="videos/coding_demo.mp4",
    relevant_paths=["documents/recallforge_spec.docx", "documents/operations_manual.pdf", "documents/edge_deployment_guide.pdf", "documents/embedding_research.pdf"],
    queries=[
        "coding demo should find RecallForge technical specification",
        "developer video connected to operations manual",
        "software walkthrough linked to edge deployment guide",
        "code demonstration should recall embedding research paper",
        "implementation video tied to install and configuration docs",
        "developer recording connected to local multimodal search spec",
    ],
    difficulty="hard",
    graded_relevance={"documents/recallforge_spec.docx": 2, "documents/operations_manual.pdf": 2, "documents/edge_deployment_guide.pdf": 1, "documents/embedding_research.pdf": 1},
)
VIDEO_TO_DOCUMENT += _media_query_variants(
    category="video_to_document",
    query_type="video",
    source_path="videos/whiteboard_session.mp4",
    relevant_paths=["documents/ai_strategy_report.docx", "documents/project_status_q1.docx", "documents/quarterly_review.pptx", "documents/recallforge_spec.docx"],
    queries=[
        "whiteboard session should retrieve strategy roadmap documents",
        "planning video connected to quarterly review presentation",
        "brainstorm recording linked to project status report",
        "whiteboard meeting should recall RecallForge spec",
        "roadmap discussion video tied to portfolio review deck",
        "team planning recording connected to AI strategy docs",
    ],
    difficulty="hard",
    graded_relevance={"documents/ai_strategy_report.docx": 2, "documents/project_status_q1.docx": 2, "documents/quarterly_review.pptx": 1, "documents/recallforge_spec.docx": 1},
)

ALL_GROUND_TRUTH = (
    TEXT_TO_TEXT + 
    TEXT_TO_IMAGE + 
    IMAGE_TO_TEXT + 
    IMAGE_TO_IMAGE +
    IMAGE_TO_VIDEO +
    IMAGE_TO_DOCUMENT +
    VIDEO_TO_TEXT +
    VIDEO_TO_IMAGE +
    VIDEO_TO_VIDEO +
    VIDEO_TO_DOCUMENT +
    MIXED_MODAL + 
    TEXT_TO_VIDEO +
    TEXT_TO_DOCUMENT +
    HARD_NEGATIVES +
    NEGATIVE_CONTROLS
)

# Export for external use
QUERIES = ALL_GROUND_TRUTH

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Results for one stage across one category."""
    stage: str
    category: str
    total_queries: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None
    hits_at_1: int = 0
    hits_at_5: int = 0
    hits_at_10: int = 0
    ndcg_sum: float = 0.0
    mrr_sum: float = 0.0
    precision_at_5_sum: float = 0.0
    precision_at_10_sum: float = 0.0
    asset_hits_at_1: int = 0
    asset_hits_at_5: int = 0
    asset_hits_at_10: int = 0
    asset_ndcg_sum: float = 0.0
    asset_mrr_sum: float = 0.0
    asset_precision_at_5_sum: float = 0.0
    asset_precision_at_10_sum: float = 0.0
    latencies_ms: list = field(default_factory=list)
    per_query_results: list = field(default_factory=list)
    
    # Per-difficulty tracking
    easy_queries: int = 0
    easy_hits_at_1: int = 0
    easy_hits_at_5: int = 0
    medium_queries: int = 0
    medium_hits_at_1: int = 0
    medium_hits_at_5: int = 0
    hard_queries: int = 0
    hard_hits_at_1: int = 0
    hard_hits_at_5: int = 0

    @property
    def recall_at_1(self) -> float:
        return self.hits_at_1 / max(self.total_queries, 1)

    @property
    def recall_at_5(self) -> float:
        return self.hits_at_5 / max(self.total_queries, 1)

    @property
    def recall_at_10(self) -> float:
        return self.hits_at_10 / max(self.total_queries, 1)

    @property
    def ndcg_at_10(self) -> float:
        return self.ndcg_sum / max(self.total_queries, 1)

    @property
    def mrr(self) -> float:
        return self.mrr_sum / max(self.total_queries, 1)
    
    @property
    def precision_at_5(self) -> float:
        return self.precision_at_5_sum / max(self.total_queries, 1)
    
    @property
    def precision_at_10(self) -> float:
        return self.precision_at_10_sum / max(self.total_queries, 1)

    @property
    def asset_recall_at_1(self) -> float:
        return self.asset_hits_at_1 / max(self.total_queries, 1)

    @property
    def asset_recall_at_5(self) -> float:
        return self.asset_hits_at_5 / max(self.total_queries, 1)

    @property
    def asset_recall_at_10(self) -> float:
        return self.asset_hits_at_10 / max(self.total_queries, 1)

    @property
    def asset_ndcg_at_10(self) -> float:
        return self.asset_ndcg_sum / max(self.total_queries, 1)

    @property
    def asset_mrr(self) -> float:
        return self.asset_mrr_sum / max(self.total_queries, 1)

    @property
    def asset_precision_at_5(self) -> float:
        return self.asset_precision_at_5_sum / max(self.total_queries, 1)

    @property
    def asset_precision_at_10(self) -> float:
        return self.asset_precision_at_10_sum / max(self.total_queries, 1)

    @property
    def p50_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]
    
    # Per-difficulty metrics
    @property
    def easy_recall_at_1(self) -> float:
        return self.easy_hits_at_1 / max(self.easy_queries, 1)
    
    @property
    def easy_recall_at_5(self) -> float:
        return self.easy_hits_at_5 / max(self.easy_queries, 1)
    
    @property
    def medium_recall_at_1(self) -> float:
        return self.medium_hits_at_1 / max(self.medium_queries, 1)
    
    @property
    def medium_recall_at_5(self) -> float:
        return self.medium_hits_at_5 / max(self.medium_queries, 1)
    
    @property
    def hard_recall_at_1(self) -> float:
        return self.hard_hits_at_1 / max(self.hard_queries, 1)
    
    @property
    def hard_recall_at_5(self) -> float:
        return self.hard_hits_at_5 / max(self.hard_queries, 1)


def _dcg(relevances: List[float], k: int = 10) -> float:
    """Discounted Cumulative Gain."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += rel / math.log2(i + 2)
    return total


def _ndcg(relevances: List[float], k: int = 10) -> float:
    """Normalized DCG."""
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


@dataclass
class EvaluationMetrics:
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    ndcg: float
    rr: float
    precision_at_5: float
    precision_at_10: float


def _normalize_benchmark_path(path: str, corpus_dir: Path) -> str:
    """Normalize benchmark filepaths to corpus-relative paths when possible."""
    raw = str(path or "").strip()
    if not raw:
        return ""

    if raw.startswith("recallforge://"):
        without_scheme = raw[len("recallforge://"):]
        _, _, raw = without_scheme.partition("/")
        raw = raw or without_scheme

    raw = re.sub(r"\s+", " ", raw).strip()
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(corpus_dir.resolve()).as_posix()
        except Exception:
            return candidate.resolve().as_posix()
    return raw.lstrip("./")


def _memory_key_for_path(path: str, corpus_dir: Path) -> str:
    """Map a result or ground-truth path to its canonical parent memory path."""
    normalized = _normalize_benchmark_path(path, corpus_dir)
    if not normalized:
        return ""
    if normalized.endswith(".transcript.json"):
        return normalized[: -len(".transcript.json")] + ".mp4"
    return normalized.split("::", 1)[0]


def _score_relevances(relevances: List[float]) -> EvaluationMetrics:
    """Compute benchmark metrics for a ranked relevance vector."""
    first_hit_rank = next((i + 1 for i, rel in enumerate(relevances[:10]) if rel > 0), None)
    hit_1 = any(rel > 0 for rel in relevances[:1])
    hit_5 = any(rel > 0 for rel in relevances[:5])
    hit_10 = any(rel > 0 for rel in relevances[:10])
    ndcg = _ndcg(relevances, 10)
    rr = 1.0 / first_hit_rank if first_hit_rank else 0.0

    max_rel = 2.0
    prec_5 = (
        sum(relevances[:5]) / (5 * max_rel)
        if len(relevances) >= 5
        else (sum(relevances) / (len(relevances) * max_rel) if relevances else 0.0)
    )
    prec_10 = (
        sum(relevances[:10]) / (10 * max_rel)
        if len(relevances) >= 10
        else (sum(relevances) / (len(relevances) * max_rel) if relevances else 0.0)
    )
    return EvaluationMetrics(
        hit_at_1=hit_1,
        hit_at_5=hit_5,
        hit_at_10=hit_10,
        ndcg=ndcg,
        rr=rr,
        precision_at_5=prec_5,
        precision_at_10=prec_10,
    )


def evaluate_results_detailed(
    results: List[Dict[str, Any]],
    gt: GroundTruth,
    corpus_dir: Path,
) -> Dict[str, EvaluationMetrics]:
    """Evaluate results at both parent-memory and raw asset granularity."""
    relevant_asset_scores: Dict[str, int] = {}
    relevant_memory_scores: Dict[str, int] = {}

    for path in gt.relevant_paths:
        normalized_path = _normalize_benchmark_path(path, corpus_dir)
        if normalized_path:
            relevant_asset_scores[normalized_path] = max(
                relevant_asset_scores.get(normalized_path, 0),
                gt.get_relevance_score(path),
            )
        memory_key = _memory_key_for_path(path, corpus_dir)
        if memory_key:
            relevant_memory_scores[memory_key] = max(
                relevant_memory_scores.get(memory_key, 0),
                gt.get_relevance_score(path),
            )

    memory_relevances: List[float] = []
    asset_relevances: List[float] = []
    for result in results[:10]:
        filepath = result.get("filepath", "")
        normalized_result = _normalize_benchmark_path(filepath, corpus_dir)
        asset_relevances.append(float(relevant_asset_scores.get(normalized_result, 0)))
        memory_key = _memory_key_for_path(filepath, corpus_dir)
        memory_relevances.append(float(relevant_memory_scores.get(memory_key, 0)))

    return {
        "memory": _score_relevances(memory_relevances),
        "asset": _score_relevances(asset_relevances),
    }


def evaluate_results(
    results: List[Dict[str, Any]],
    gt: GroundTruth,
    corpus_dir: Path,
) -> Tuple[bool, bool, bool, float, float, float, float]:
    """Evaluate search results against ground truth with graded relevance.
    
    Returns: (hit@1, hit@5, hit@10, ndcg@10, reciprocal_rank, precision@5, precision@10)
    """
    memory_metrics = evaluate_results_detailed(results, gt, corpus_dir)["memory"]
    return (
        memory_metrics.hit_at_1,
        memory_metrics.hit_at_5,
        memory_metrics.hit_at_10,
        memory_metrics.ndcg,
        memory_metrics.rr,
        memory_metrics.precision_at_5,
        memory_metrics.precision_at_10,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def ingest_corpus(backend, storage, collection: str, corpus_dir: Path) -> int:
    """Ingest the full UAT corpus into RecallForge."""
    indexed = 0

    # Text files
    for txt_file in sorted((corpus_dir / "text").glob("*.md")):
        text = txt_file.read_text(encoding="utf-8")
        try:
            storage.index_document(
                path=str(txt_file),
                text=text,
                collection=collection,
                model="benchmark",
                embed_func=backend.embed_text,
            )
            indexed += 1
        except Exception as e:
            print(f"  WARN: Failed to index {txt_file.name}: {e}")

    # Images
    for img_file in sorted((corpus_dir / "images").glob("*.png")):
        try:
            storage.index_image(
                path=str(img_file),
                collection=collection,
                embed_func=backend.embed_image,
                model="benchmark",
            )
            indexed += 1
        except Exception as e:
            print(f"  WARN: Failed to index {img_file.name}: {e}")

    # Documents (generated)
    doc_dir = corpus_dir / "documents"
    if doc_dir.exists():
        for doc_file in sorted(doc_dir.iterdir()):
            if doc_file.suffix in (".docx", ".pptx", ".pdf"):
                try:
                    storage.index_document_file(
                        path=str(doc_file),
                        collection=collection,
                        embed_func=backend.embed_text,
                        embed_image_func=backend.embed_image,
                        model="benchmark",
                    )
                    indexed += 1
                except Exception as e:
                    print(f"  WARN: Failed to index {doc_file.name}: {e}")
    
    # Videos and transcripts
    video_dir = corpus_dir / "videos"
    if video_dir.exists():
        for video_file in sorted(video_dir.glob("*.mp4")):
            try:
                # Index video
                storage.index_video(
                    path=str(video_file),
                    collection=collection,
                    embed_text_func=backend.embed_text,
                    embed_image_func=backend.embed_image,
                    embed_video_func=backend.embed_video,
                    model="benchmark",
                )
                indexed += 1
            except Exception as e:
                print(f"  WARN: Failed to index video {video_file.name}: {e}")
            # Reclaim Metal memory between videos to avoid OOM on 16GB devices
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
        
        # Index transcript JSONs
        for transcript_file in sorted(video_dir.glob("*.transcript.json")):
            try:
                with open(transcript_file) as f:
                    transcript_data = json.load(f)
                    text = transcript_data.get("text", "")
                    if text:
                        storage.index_document(
                            path=str(transcript_file),
                            text=text,
                            collection=collection,
                            model="benchmark",
                            embed_func=backend.embed_text,
                        )
                        indexed += 1
            except Exception as e:
                print(f"  WARN: Failed to index transcript {transcript_file.name}: {e}")

    # Unload captioner after indexing to reclaim ~0.9GB before search stages
    if hasattr(backend, '_unload_captioner'):
        backend._unload_captioner()

    return indexed


STAGES = [
    ("Vector-only", "embed"),
    ("BM25-only", "bm25"),
    ("Vector + BM25 (RRF)", "rrf"),
    ("Vector + BM25 + Reranker", "hybrid"),
]


@dataclass(frozen=True)
class ExpansionProfile:
    """Benchmark-time query expansion configuration."""

    name: str
    expand: bool
    enable_media_query_probe: bool
    allow_generate_text: bool


EXPANSION_PROFILES: Dict[str, ExpansionProfile] = {
    # Text queries: no expansion.
    # Media queries: pure vector search with no caption/transcript BM25 probe.
    "off": ExpansionProfile(
        name="off",
        expand=False,
        enable_media_query_probe=False,
        allow_generate_text=False,
    ),
    # Text queries: no expansion.
    # Media queries: keep caption/transcript BM25 probe, but do not add expansion branches.
    "caption_only": ExpansionProfile(
        name="caption_only",
        expand=False,
        enable_media_query_probe=True,
        allow_generate_text=False,
    ),
    # Text/media probe expansion uses heuristic fallback rules only.
    "heuristic": ExpansionProfile(
        name="heuristic",
        expand=True,
        enable_media_query_probe=True,
        allow_generate_text=False,
    ),
    # Text/media probe expansion uses the backend's generator when available.
    "qwen": ExpansionProfile(
        name="qwen",
        expand=True,
        enable_media_query_probe=True,
        allow_generate_text=True,
    ),
}


class _ExpansionBackendProxy:
    """Optionally hide generated expansion so benchmarks can force heuristics."""

    def __init__(self, backend, *, allow_generate_text: bool):
        self._backend = backend
        self._allow_generate_text = allow_generate_text

    def generate_text(self, prompt: str, max_tokens: int = 60) -> str:
        if not self._allow_generate_text:
            raise NotImplementedError(
                "generate_text disabled for this benchmark expansion profile"
            )
        generator = getattr(self._backend, "generate_text", None)
        if not callable(generator):
            raise NotImplementedError("Underlying backend does not support generate_text")
        return generator(prompt, max_tokens=max_tokens)

    def __getattr__(self, name: str):
        return getattr(self._backend, name)


def _resolve_expansion_profile(profile_name: str) -> ExpansionProfile:
    """Return the named query-expansion benchmark profile."""
    try:
        return EXPANSION_PROFILES[profile_name]
    except KeyError as exc:
        valid = ", ".join(sorted(EXPANSION_PROFILES))
        raise ValueError(
            f"Unknown expansion profile: {profile_name}. Valid choices: {valid}"
        ) from exc


def _group_queries(
    category_filters: Optional[List[str]] = None,
    max_queries_per_category: Optional[int] = None,
) -> Dict[str, List["GroundTruth"]]:
    """Group benchmark queries with optional category and per-category limits."""
    requested = set(category_filters or [])
    categories: Dict[str, List[GroundTruth]] = {}

    for gt in ALL_GROUND_TRUTH:
        if requested and gt.category not in requested:
            continue
        bucket = categories.setdefault(gt.category, [])
        if max_queries_per_category is None or len(bucket) < max_queries_per_category:
            bucket.append(gt)

    if requested:
        missing = sorted(requested - set(categories.keys()))
        if missing:
            raise ValueError(f"Unknown categories requested: {', '.join(missing)}")

    return categories


def _select_stages(stage_filters: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Return the selected benchmark stages by mode or display name."""
    if not stage_filters:
        return STAGES

    requested = set(stage_filters)
    selected = [
        (stage_name, stage_mode)
        for stage_name, stage_mode in STAGES
        if stage_mode in requested or stage_name in requested
    ]
    if not selected:
        raise ValueError(f"Unknown stages requested: {', '.join(sorted(requested))}")
    return selected


def _resolve_output_path(output_path: Optional[str], expansion_profile: str) -> str:
    """Return the benchmark output path, falling back to the default results file."""
    if output_path:
        return output_path

    suffix = "" if expansion_profile == "caption_only" else f"_{expansion_profile}"
    return str(
        PROJECT_ROOT
        / "benchmarks"
        / "results"
        / f"cross_modal_ablation_results{suffix}.json"
    )


def _apply_smoke_profile_defaults(
    smoke_profile: str,
    stage_filters: Optional[List[str]],
    max_queries_per_category: Optional[int],
    rss_limit_mb: Optional[int],
) -> tuple[Optional[List[str]], Optional[int], Optional[int]]:
    """Resolve smoke-profile defaults without overriding explicit caller choices."""
    if smoke_profile != "safe":
        return stage_filters, max_queries_per_category, rss_limit_mb

    resolved_stage_filters = stage_filters or ["rrf"]
    resolved_max_queries = max_queries_per_category or 1
    resolved_rss_limit = rss_limit_mb or 6144
    return resolved_stage_filters, resolved_max_queries, resolved_rss_limit


def _current_rss_mb() -> Optional[float]:
    """Return current peak RSS in MB when available."""
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024.0


def _build_output_payload(
    categories: Dict[str, List[GroundTruth]],
    all_results: Dict[str, Dict[str, StageResult]],
    stages: List[Tuple[str, str]],
    *,
    expansion_profile: ExpansionProfile,
    smoke_profile: str,
    rss_limit_mb: Optional[int],
    peak_rss_mb: Optional[float],
    indexed_items: int,
    run_status: str,
    interrupted: bool,
    completed_stages: List[str],
    current_stage: Optional[str],
    current_category: Optional[str],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize benchmark progress/results into the published JSON payload."""
    output = {
        "benchmark": "cross_modal_ablation",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "expansion_profile": expansion_profile.name,
            "expand_enabled": expansion_profile.expand,
            "media_query_probe_enabled": expansion_profile.enable_media_query_probe,
            "generated_query_expansion_enabled": expansion_profile.allow_generate_text,
            "smoke_profile": smoke_profile,
            "rss_limit_mb": rss_limit_mb,
        },
        "run_status": run_status,
        "interrupted": interrupted,
        "progress": {
            "indexed_items": indexed_items,
            "completed_stages": completed_stages,
            "completed_categories": {
                stage_name: sorted(stage_results.keys())
                for stage_name, stage_results in all_results.items()
            },
            "current_stage": current_stage,
            "current_category": current_category,
        },
        "telemetry": {
            "peak_rss_mb": None if peak_rss_mb is None else round(peak_rss_mb, 1),
        },
        "corpus": {
            "text_docs": len(list((CORPUS_DIR / "text").glob("*.md"))),
            "images": len(list((CORPUS_DIR / "images").glob("*.png"))),
            "videos": len(list((CORPUS_DIR / "videos").glob("*.mp4"))),
            "total_corpus_docs": len(CORPUS_DOCS),
            "total_queries": sum(len(v) for v in categories.values()),
            "queries_by_category": {k: len(v) for k, v in categories.items()},
        },
        "categories": {},
        "stages": {},
    }
    if error:
        output["error"] = error

    for cat_name, queries in categories.items():
        output["categories"][cat_name] = {
            "queries": len(queries),
            "by_difficulty": {
                "easy": sum(1 for q in queries if q.difficulty == "easy"),
                "medium": sum(1 for q in queries if q.difficulty == "medium"),
                "hard": sum(1 for q in queries if q.difficulty == "hard"),
            }
        }

    for stage_name, _ in stages:
        output["stages"][stage_name] = {}
        for cat_name, sr in all_results.get(stage_name, {}).items():
            output["stages"][stage_name][cat_name] = {
                "skipped": sr.skipped,
                "skip_reason": sr.skip_reason,
                "recall_at_1": None if sr.skipped else round(sr.recall_at_1, 4),
                "recall_at_5": None if sr.skipped else round(sr.recall_at_5, 4),
                "recall_at_10": None if sr.skipped else round(sr.recall_at_10, 4),
                "precision_at_5": None if sr.skipped else round(sr.precision_at_5, 4),
                "precision_at_10": None if sr.skipped else round(sr.precision_at_10, 4),
                "ndcg_at_10": None if sr.skipped else round(sr.ndcg_at_10, 4),
                "mrr": None if sr.skipped else round(sr.mrr, 4),
                "p50_ms": None if sr.skipped else round(sr.p50_ms, 1),
                "p95_ms": None if sr.skipped else round(sr.p95_ms, 1),
                "asset_level": {
                    "recall_at_1": None if sr.skipped else round(sr.asset_recall_at_1, 4),
                    "recall_at_5": None if sr.skipped else round(sr.asset_recall_at_5, 4),
                    "recall_at_10": None if sr.skipped else round(sr.asset_recall_at_10, 4),
                    "precision_at_5": None if sr.skipped else round(sr.asset_precision_at_5, 4),
                    "precision_at_10": None if sr.skipped else round(sr.asset_precision_at_10, 4),
                    "ndcg_at_10": None if sr.skipped else round(sr.asset_ndcg_at_10, 4),
                    "mrr": None if sr.skipped else round(sr.asset_mrr, 4),
                },
                "total_queries": sr.total_queries,
                "by_difficulty": {
                    "easy": {
                        "queries": sr.easy_queries,
                        "recall_at_1": None if sr.skipped or sr.easy_queries == 0 else round(sr.easy_recall_at_1, 4),
                        "recall_at_5": None if sr.skipped or sr.easy_queries == 0 else round(sr.easy_recall_at_5, 4),
                    },
                    "medium": {
                        "queries": sr.medium_queries,
                        "recall_at_1": None if sr.skipped or sr.medium_queries == 0 else round(sr.medium_recall_at_1, 4),
                        "recall_at_5": None if sr.skipped or sr.medium_queries == 0 else round(sr.medium_recall_at_5, 4),
                    },
                    "hard": {
                        "queries": sr.hard_queries,
                        "recall_at_1": None if sr.skipped or sr.hard_queries == 0 else round(sr.hard_recall_at_1, 4),
                        "recall_at_5": None if sr.skipped or sr.hard_queries == 0 else round(sr.hard_recall_at_5, 4),
                    },
                },
                "per_query_results": sr.per_query_results,
            }

    return output


def _write_output_payload(output: Dict[str, Any], save_path: str) -> None:
    """Write output JSON atomically so interruptions do not leave a partial file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_path = f"{save_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(temp_path, save_path)


def _skip_reason_for_stage(stage_mode: str, queries: List[GroundTruth]) -> Optional[str]:
    """Return a reason when a stage cannot support a category's query modality."""
    if stage_mode != "bm25" or not queries:
        return None

    query_types = {gt.query_type for gt in queries}
    if "text" in query_types:
        return None
    if query_types == {"image"}:
        return "BM25 can't process image queries"
    if query_types == {"video"}:
        return "BM25 can't process video queries"
    return "BM25 can't process image/video queries"


def _result_content_type_for_category(category: str) -> Optional[str]:
    """Return the expected result modality for category-specific benchmark runs."""
    mapping = {
        "text_to_text": "text",
        "text_to_image": "image",
        "text_to_video": "video",
        "image_to_text": "text",
        "image_to_image": "image",
        "image_to_video": "video",
        "video_to_text": "text",
        "video_to_image": "image",
        "video_to_video": "video",
    }
    return mapping.get(category)


def run_search(
    backend,
    storage,
    gt: GroundTruth,
    collection: str,
    stage_mode: str,
    limit: int = 10,
    expansion_profile: str = "caption_only",
) -> Tuple[List[Dict], float]:
    """Run a single search query and return results + latency_ms."""
    from recallforge.search import HybridSearcher

    t0 = time.perf_counter()
    result_content_type = _result_content_type_for_category(gt.category)
    profile = _resolve_expansion_profile(expansion_profile)
    search_backend = _ExpansionBackendProxy(
        backend,
        allow_generate_text=profile.allow_generate_text,
    )

    if gt.query_type == "image" and gt.image_query_path:
        image_path = str(CORPUS_DIR / gt.image_query_path)

        if stage_mode == "embed":
            image_vec = backend.embed_image(image_path)
            results = storage.search_vec(
                vector=image_vec,
                collection=collection,
                content_type=result_content_type,
                limit=limit,
            )
        elif stage_mode in ("rrf", "hybrid"):
            searcher = HybridSearcher(
                backend=search_backend,
                storage=storage,
                limit=limit,
                collection=collection,
                content_type=result_content_type,
                expand=profile.expand,
                enable_media_query_probe=profile.enable_media_query_probe,
            )
            old_mode = search_backend.get_mode()
            search_backend.set_mode("embed" if stage_mode == "rrf" else "hybrid")
            results = searcher.search_image(image_path)
            search_backend.set_mode(old_mode)
        elif stage_mode == "bm25":
            results = []
        else:
            raise ValueError(f"Unknown stage mode: {stage_mode}")

    elif gt.query_type == "video" and gt.video_query_path:
        video_path = str(CORPUS_DIR / gt.video_query_path)

        if stage_mode == "embed":
            video_vec = backend.embed_video(video_path)
            results = storage.search_vec(
                vector=video_vec,
                collection=collection,
                content_type=result_content_type,
                limit=limit,
            )
        elif stage_mode in ("rrf", "hybrid"):
            searcher = HybridSearcher(
                backend=search_backend,
                storage=storage,
                limit=limit,
                collection=collection,
                content_type=result_content_type,
                expand=profile.expand,
                enable_media_query_probe=profile.enable_media_query_probe,
            )
            old_mode = search_backend.get_mode()
            search_backend.set_mode("embed" if stage_mode == "rrf" else "hybrid")
            results = searcher.search_video(video_path)
            search_backend.set_mode(old_mode)
        elif stage_mode == "bm25":
            results = []
        else:
            raise ValueError(f"Unknown stage mode: {stage_mode}")

    else:
        # Text query
        if stage_mode == "embed":
            # Vector only
            query_vec = backend.embed_text(gt.query)
            results = storage.search_vec(
                vector=query_vec,
                collection=collection,
                content_type=result_content_type,
                limit=limit,
            )
        elif stage_mode == "bm25":
            # BM25 only
            results = storage.search_fts(
                query=gt.query,
                collection=collection,
                content_type=result_content_type,
                limit=limit,
            )
        elif stage_mode == "rrf":
            # RRF without reranker (embed mode)
            searcher = HybridSearcher(
                backend=search_backend,
                storage=storage,
                limit=limit,
                collection=collection,
                content_type=result_content_type,
                expand=profile.expand,
                enable_media_query_probe=profile.enable_media_query_probe,
            )
            old_mode = search_backend.get_mode()
            search_backend.set_mode("embed")
            results = searcher.search(gt.query)
            search_backend.set_mode(old_mode)
        elif stage_mode == "hybrid":
            # Full hybrid with reranker
            searcher = HybridSearcher(
                backend=search_backend,
                storage=storage,
                limit=limit,
                collection=collection,
                content_type=result_content_type,
                expand=profile.expand,
                enable_media_query_probe=profile.enable_media_query_probe,
            )
            old_mode = search_backend.get_mode()
            search_backend.set_mode("hybrid")
            results = searcher.search(gt.query)
            search_backend.set_mode(old_mode)
        else:
            raise ValueError(f"Unknown stage mode: {stage_mode}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Normalize results to dicts and capture audit trail
    result_dicts = []
    for r in results:
        if isinstance(r, dict):
            result_dicts.append(r)
        else:
            result = {
                "filepath": getattr(r, "filepath", getattr(r, "path", "")),
                "title": getattr(r, "title", ""),
                "score": getattr(r, "score", 0.0),
            }
            # Capture audit trail if available (HybridResult has audit)
            audit = getattr(r, "audit", None)
            if audit is not None:
                result["audit"] = {
                    "filepath": audit.filepath,
                    "content_type": audit.content_type,
                    "rrf_sources": audit.rrf_sources,
                    "reranker_raw_score": audit.reranker_raw_score,
                    "reranker_normalized_score": audit.reranker_normalized_score,
                    "reranker_scoring_path": audit.reranker_scoring_path,
                    "blend_weights": audit.blend_weights,
                    "final_blended_score": audit.final_blended_score,
                }
            result_dicts.append(result)

    return result_dicts, elapsed_ms


def run_benchmark(
    backend,
    storage,
    collection: str = "benchmark",
    output_path: Optional[str] = None,
    category_filters: Optional[List[str]] = None,
    stage_filters: Optional[List[str]] = None,
    max_queries_per_category: Optional[int] = None,
    expansion_profile: str = "caption_only",
    smoke_profile: str = "off",
    rss_limit_mb: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full cross-modal ablation benchmark."""

    stage_filters, max_queries_per_category, rss_limit_mb = _apply_smoke_profile_defaults(
        smoke_profile,
        stage_filters,
        max_queries_per_category,
        rss_limit_mb,
    )

    categories = _group_queries(
        category_filters=category_filters,
        max_queries_per_category=max_queries_per_category,
    )
    if not categories:
        raise ValueError("No benchmark queries selected")
    stages = _select_stages(stage_filters)
    profile = _resolve_expansion_profile(expansion_profile)
    save_path = _resolve_output_path(output_path, profile.name)
    indexed = 0
    completed_stages: List[str] = []
    current_stage_name: Optional[str] = None
    current_category_name: Optional[str] = None
    peak_rss_mb = _current_rss_mb()

    def save_checkpoint(
        *,
        run_status: str,
        interrupted: bool = False,
        error: Optional[str] = None,
        announce: bool = False,
    ) -> Dict[str, Any]:
        output = _build_output_payload(
            categories,
            all_results,
            stages,
            expansion_profile=profile,
            smoke_profile=smoke_profile,
            rss_limit_mb=rss_limit_mb,
            peak_rss_mb=peak_rss_mb,
            indexed_items=indexed,
            run_status=run_status,
            interrupted=interrupted,
            completed_stages=completed_stages.copy(),
            current_stage=current_stage_name,
            current_category=current_category_name,
            error=error,
        )
        _write_output_payload(output, save_path)
        if announce:
            message = "Saved partial results" if interrupted or run_status != "complete" else "Saved"
            print(f"\n{message}: {save_path}")
        return output

    print(f"\nQuery categories: {', '.join(f'{k}({len(v)})' for k, v in categories.items())}")
    print(f"Stages: {', '.join(stage_name for stage_name, _ in stages)}")
    print(f"Expansion profile: {profile.name}")
    print(f"Smoke profile: {smoke_profile}")
    if rss_limit_mb is not None:
        print(f"RSS limit: {rss_limit_mb} MB")
    print(f"Total queries: {sum(len(v) for v in categories.values())}")
    print(f"Corpus documents: {len(CORPUS_DOCS)}")

    all_results: Dict[str, Dict[str, StageResult]] = {}

    try:
        # Ingest corpus
        print("\nIndexing corpus...")
        indexed = ingest_corpus(backend, storage, collection, CORPUS_DIR)
        peak_rss_mb = max(peak_rss_mb or 0.0, _current_rss_mb() or 0.0)
        if rss_limit_mb is not None and peak_rss_mb and peak_rss_mb > rss_limit_mb:
            raise RuntimeError(
                f"RSS limit exceeded during indexing: peak {peak_rss_mb:.1f} MB > limit {rss_limit_mb} MB"
            )
        print(f"Indexed {indexed} items.\n")
        save_checkpoint(run_status="partial")

        # Run all stages × categories
        for stage_name, stage_mode in stages:
            current_stage_name = stage_name
            current_category_name = None
            all_results[stage_name] = {}

            # Clear Metal cache between stages to avoid OOM on 16GB devices
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

            for cat_name, queries in categories.items():
                current_category_name = cat_name
                sr = StageResult(stage=stage_name, category=cat_name, total_queries=len(queries))

                # Track per-difficulty counts
                for gt in queries:
                    if gt.difficulty == "easy":
                        sr.easy_queries += 1
                    elif gt.difficulty == "medium":
                        sr.medium_queries += 1
                    elif gt.difficulty == "hard":
                        sr.hard_queries += 1

                skip_reason = _skip_reason_for_stage(stage_mode, queries)
                if skip_reason:
                    sr.skipped = True
                    sr.skip_reason = skip_reason
                    all_results[stage_name][cat_name] = sr
                    print(f"  {stage_name} for {cat_name}: SKIPPED ({skip_reason})")
                    save_checkpoint(run_status="partial")
                    continue

                effective_mode = stage_mode

                for gt in queries:
                    try:
                        results, latency = run_search(
                            backend, storage, gt,
                            collection, effective_mode,
                            expansion_profile=profile.name,
                        )
                        eval_detail = evaluate_results_detailed(results, gt, CORPUS_DIR)
                        memory_metrics = eval_detail["memory"]
                        asset_metrics = eval_detail["asset"]

                        sr.hits_at_1 += int(memory_metrics.hit_at_1)
                        sr.hits_at_5 += int(memory_metrics.hit_at_5)
                        sr.hits_at_10 += int(memory_metrics.hit_at_10)
                        sr.ndcg_sum += memory_metrics.ndcg
                        sr.mrr_sum += memory_metrics.rr
                        sr.precision_at_5_sum += memory_metrics.precision_at_5
                        sr.precision_at_10_sum += memory_metrics.precision_at_10
                        sr.asset_hits_at_1 += int(asset_metrics.hit_at_1)
                        sr.asset_hits_at_5 += int(asset_metrics.hit_at_5)
                        sr.asset_hits_at_10 += int(asset_metrics.hit_at_10)
                        sr.asset_ndcg_sum += asset_metrics.ndcg
                        sr.asset_mrr_sum += asset_metrics.rr
                        sr.asset_precision_at_5_sum += asset_metrics.precision_at_5
                        sr.asset_precision_at_10_sum += asset_metrics.precision_at_10
                        sr.latencies_ms.append(latency)
                        peak_rss_mb = max(peak_rss_mb or 0.0, _current_rss_mb() or 0.0)
                        if rss_limit_mb is not None and peak_rss_mb and peak_rss_mb > rss_limit_mb:
                            raise RuntimeError(
                                f"RSS limit exceeded: peak {peak_rss_mb:.1f} MB > limit {rss_limit_mb} MB"
                            )

                        # Track per-difficulty hits
                        if gt.difficulty == "easy":
                            sr.easy_hits_at_1 += int(memory_metrics.hit_at_1)
                            sr.easy_hits_at_5 += int(memory_metrics.hit_at_5)
                        elif gt.difficulty == "medium":
                            sr.medium_hits_at_1 += int(memory_metrics.hit_at_1)
                            sr.medium_hits_at_5 += int(memory_metrics.hit_at_5)
                        elif gt.difficulty == "hard":
                            sr.hard_hits_at_1 += int(memory_metrics.hit_at_1)
                            sr.hard_hits_at_5 += int(memory_metrics.hit_at_5)

                        # Store per-query result with audit trail for post-hoc analysis
                        sr.per_query_results.append({
                            "query": gt.query,
                            "query_type": gt.query_type,
                            "image_query_path": gt.image_query_path,
                            "video_query_path": gt.video_query_path,
                            "relevant_paths": gt.relevant_paths,
                            "difficulty": gt.difficulty,
                            "is_negative_control": gt.is_negative_control,
                            "hit_at_1": memory_metrics.hit_at_1,
                            "hit_at_5": memory_metrics.hit_at_5,
                            "hit_at_10": memory_metrics.hit_at_10,
                            "ndcg": memory_metrics.ndcg,
                            "mrr": memory_metrics.rr,
                            "precision_at_5": memory_metrics.precision_at_5,
                            "precision_at_10": memory_metrics.precision_at_10,
                            "asset_level": {
                                "hit_at_1": asset_metrics.hit_at_1,
                                "hit_at_5": asset_metrics.hit_at_5,
                                "hit_at_10": asset_metrics.hit_at_10,
                                "ndcg": asset_metrics.ndcg,
                                "mrr": asset_metrics.rr,
                                "precision_at_5": asset_metrics.precision_at_5,
                                "precision_at_10": asset_metrics.precision_at_10,
                            },
                            "latency_ms": latency,
                            "results": results,
                        })
                    except Exception as e:
                        print(f"    ERROR: {gt.query[:50]}... → {e}")
                        sr.latencies_ms.append(0)
                        sr.per_query_results.append({
                            "query": gt.query,
                            "query_type": gt.query_type,
                            "image_query_path": gt.image_query_path,
                            "video_query_path": gt.video_query_path,
                            "relevant_paths": gt.relevant_paths,
                            "difficulty": gt.difficulty,
                            "is_negative_control": gt.is_negative_control,
                            "error": str(e),
                        })

                all_results[stage_name][cat_name] = sr
                print(f"  {stage_name} for {cat_name} ({len(queries)}q)... "
                      f"R@1={sr.recall_at_1:.1%} R@5={sr.recall_at_5:.1%} "
                      f"R@10={sr.recall_at_10:.1%} AssetR@1={sr.asset_recall_at_1:.1%} P@5={sr.precision_at_5:.3f} "
                      f"NDCG@10={sr.ndcg_at_10:.3f} MRR={sr.mrr:.3f}")
                save_checkpoint(run_status="partial")

            completed_stages.append(stage_name)
            current_category_name = None
            save_checkpoint(run_status="partial")

    except KeyboardInterrupt:
        save_checkpoint(
            run_status="partial",
            interrupted=True,
            error="Interrupted by user",
            announce=True,
        )
        raise

    current_stage_name = None
    current_category_name = None
    complete_output = save_checkpoint(run_status="complete", announce=True)

    # Print results tables
    print("\n" + "=" * 120)
    for cat_name in categories:
        print(f"\n{'=' * 120}")
        print(f"  {cat_name.upper()} ({len(categories[cat_name])} queries)")
        print(f"{'=' * 120}")
        print(f"{'Stage':<35} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'P@5':>7} {'P@10':>7} "
              f"{'NDCG@10':>8} {'MRR':>6} {'p50':>8} {'p95':>8}")
        print("-" * 120)

        for stage_name, _ in stages:
            sr = all_results[stage_name][cat_name]
            if sr.total_queries == 0:
                continue
            if sr.skipped:
                print(f"{stage_name:<35} SKIPPED ({sr.skip_reason})")
                continue
            print(f"{stage_name:<35} "
                  f"{sr.recall_at_1:>5.1%} {sr.recall_at_5:>5.1%} {sr.recall_at_10:>5.1%} "
                  f"{sr.precision_at_5:>6.3f} {sr.precision_at_10:>6.3f} "
                  f"{sr.ndcg_at_10:>7.3f} {sr.mrr:>5.3f} "
                  f"{sr.p50_ms:>7.0f}ms {sr.p95_ms:>7.0f}ms")

    # Per-difficulty breakdown
    print(f"\n{'=' * 120}")
    print("  DIFFICULTY BREAKDOWN (R@1 / R@5)")
    print(f"{'=' * 120}")
    print(f"{'Category':<20} {'Stage':<30} {'Easy':>12} {'Medium':>12} {'Hard':>12}")
    print("-" * 100)

    for cat_name in categories:
        for stage_name, _ in stages:
            sr = all_results[stage_name][cat_name]
            if sr.total_queries == 0:
                continue
            if sr.skipped:
                print(f"{cat_name:<20} {stage_name:<30} {'SKIPPED':>12}")
                continue
            easy_str = f"{sr.easy_recall_at_1:.1%}/{sr.easy_recall_at_5:.1%}" if sr.easy_queries > 0 else "N/A"
            med_str = f"{sr.medium_recall_at_1:.1%}/{sr.medium_recall_at_5:.1%}" if sr.medium_queries > 0 else "N/A"
            hard_str = f"{sr.hard_recall_at_1:.1%}/{sr.hard_recall_at_5:.1%}" if sr.hard_queries > 0 else "N/A"
            print(f"{cat_name:<20} {stage_name:<30} {easy_str:>12} {med_str:>12} {hard_str:>12}")

    # Reranker uplift summary
    print(f"\n{'=' * 120}")
    print("  RERANKER UPLIFT SUMMARY (R@1 and NDCG@10)")
    print(f"{'=' * 120}")
    print(f"{'Category':<20} {'Vec R@1':>8} {'Rerank R@1':>10} {'Δ R@1':>8} "
          f"{'Vec NDCG':>9} {'Rerank NDCG':>12} {'Δ NDCG':>8}")
    print("-" * 90)

    for cat_name in categories:
        vec = all_results.get("Vector-only", {}).get(cat_name)
        rerank = all_results.get("Vector + BM25 + Reranker", {}).get(cat_name)
        if vec and rerank and vec.total_queries > 0:
            delta_r1 = rerank.recall_at_1 - vec.recall_at_1
            delta_ndcg = rerank.ndcg_at_10 - vec.ndcg_at_10
            print(f"{cat_name:<20} "
                  f"{vec.recall_at_1:>7.1%} {rerank.recall_at_1:>9.1%} "
                  f"{delta_r1:>+7.1%} "
                  f"{vec.ndcg_at_10:>8.3f} {rerank.ndcg_at_10:>11.3f} "
                  f"{delta_ndcg:>+7.3f}")

    return complete_output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-modal pipeline ablation benchmark"
    )
    parser.add_argument(
        "--backend", choices=["torch", "mlx", "auto"], default="auto",
        help="Model backend",
    )
    parser.add_argument(
        "--quantization", default="4bit",
        help="MLX quantization level",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--store-path", default=None,
        help="Storage directory (default: temp dir)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate query structure without running benchmark",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Benchmark category to run (repeatable)",
    )
    parser.add_argument(
        "--stage-mode",
        action="append",
        choices=["embed", "bm25", "rrf", "hybrid"],
        default=None,
        help="Benchmark stage mode to run (repeatable)",
    )
    parser.add_argument(
        "--max-queries-per-category",
        type=int,
        default=None,
        help="Cap how many queries to run from each selected category",
    )
    parser.add_argument(
        "--expansion-profile",
        choices=sorted(EXPANSION_PROFILES),
        default="caption_only",
        help=(
            "Query expansion profile: off (pure vector/media), caption_only "
            "(current default), heuristic, or qwen"
        ),
    )
    parser.add_argument(
        "--smoke-profile",
        choices=["off", "safe"],
        default="off",
        help="Optional bounded smoke profile for safer local validation.",
    )
    parser.add_argument(
        "--rss-limit-mb",
        type=int,
        default=None,
        help="Abort the benchmark if peak RSS exceeds this limit in MB.",
    )
    args = parser.parse_args()

    if args.dry_run:
        resolved_stage_mode, resolved_max_queries, resolved_rss_limit = _apply_smoke_profile_defaults(
            args.smoke_profile,
            args.stage_mode,
            args.max_queries_per_category,
            args.rss_limit_mb,
        )
        # Validate query structure
        print(f"\n{'=' * 60}")
        print("DRY RUN - Query Structure Validation")
        print(f"{'=' * 60}")
        categories = _group_queries(
            category_filters=args.category,
            max_queries_per_category=resolved_max_queries,
        )
        print(f"Total queries: {sum(len(v) for v in categories.values())}")
        print(f"Corpus documents: {len(CORPUS_DOCS)}")
        print(f"Expansion profile: {args.expansion_profile}")
        print(f"Smoke profile: {args.smoke_profile}")
        print(f"Resolved stages: {resolved_stage_mode or 'all'}")
        if resolved_rss_limit is not None:
            print(f"Resolved RSS limit: {resolved_rss_limit} MB")

        print(f"\nQueries by category:")
        for cat, queries in categories.items():
            easy = sum(1 for q in queries if q.difficulty == "easy")
            medium = sum(1 for q in queries if q.difficulty == "medium")
            hard = sum(1 for q in queries if q.difficulty == "hard")
            neg = sum(1 for q in queries if q.is_negative_control)
            print(f"  {cat}: {len(queries)} (easy={easy}, medium={medium}, hard={hard}, negative={neg})")
        
        # Validate all relevant_paths exist in corpus
        missing = []
        for queries in categories.values():
            for gt in queries:
                for path in gt.relevant_paths:
                    if path not in CORPUS_DOCS:
                        missing.append((gt.query[:50], path))
        
        if missing:
            print(f"\nWARNING: {len(missing)} paths not in CORPUS_DOCS:")
            for q, p in missing[:10]:
                print(f"  Query '{q}...' -> {p}")
        else:
            print("\n✓ All relevant_paths validated against CORPUS_DOCS")
        
        print(f"\n{'=' * 60}")
        print("Validation complete")
        print(f"{'=' * 60}")
        return

    # Initialize backend
    from recallforge import get_backend, get_storage

    store_path = args.store_path or tempfile.mkdtemp(prefix="recallforge_bench_")
    cleanup = args.store_path is None

    try:
        os.environ["RECALLFORGE_BACKEND"] = args.backend
        if args.quantization:
            os.environ["RECALLFORGE_MLX_QUANTIZE"] = args.quantization

        backend = get_backend()
        storage = get_storage(store_path)
        storage.initialize(store_path)

        info = backend.get_info()
        print(f"Backend: {info.name}")
        print(f"Device: {info.device}")
        print(f"Quantization: {info.quantization}")
        print(f"Store: {store_path}")

        run_benchmark(
            backend,
            storage,
            output_path=args.output,
            category_filters=args.category,
            stage_filters=args.stage_mode,
            max_queries_per_category=args.max_queries_per_category,
            expansion_profile=args.expansion_profile,
            smoke_profile=args.smoke_profile,
            rss_limit_mb=args.rss_limit_mb,
        )

    finally:
        if cleanup and os.path.exists(store_path):
            shutil.rmtree(store_path, ignore_errors=True)


if __name__ == "__main__":
    main()
