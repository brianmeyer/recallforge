"""Tests for lightweight entity and relation extraction."""

from recallforge.entities import (
    extract_entities,
    extract_relations,
    normalize_entity_key,
    stable_graph_id,
)


def test_extract_entities_deduplicates_and_classifies_common_memory_entities():
    text = (
        "Alice from Acme Robotics discussed REC-76 with @brian. "
        "The notes live at https://recallforge.dev/roadmap."
    )

    entities = extract_entities(text)
    by_key = {entity.entity_key: entity for entity in entities}

    assert "alice" in by_key
    assert by_key["alice"].entity_type == "proper_noun"
    assert "acme_robotics" in by_key
    assert "rec_76" in by_key
    assert by_key["rec_76"].entity_type == "ticket"
    assert "brian" in by_key
    assert by_key["brian"].entity_type == "person"
    assert "https_recallforge_dev_roadmap" in by_key
    assert by_key["https_recallforge_dev_roadmap"].entity_type == "url"
    assert "Acme Robotics" in by_key["acme_robotics"].evidence


def test_extract_relations_creates_traceable_co_mentions():
    entities = extract_entities("Alice and Bob met at Acme Robotics for Project Atlas.")

    relations = extract_relations(entities)

    assert relations
    assert any(
        relation.relation_type == "co_mentions"
        and {relation.subject_key, relation.object_key} >= {"alice", "bob"}
        for relation in relations
    )
    assert all(relation.evidence for relation in relations)


def test_normalize_entity_key_and_stable_graph_id_are_deterministic():
    assert normalize_entity_key("@Brian Meyer") == "brian_meyer"
    assert stable_graph_id("entity", "hash_0", "acme") == stable_graph_id("entity", "hash_0", "acme")
    assert stable_graph_id("entity", "hash_0", "acme") != stable_graph_id("entity", "hash_1", "acme")
