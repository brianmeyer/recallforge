# Corpus Expansion Documentation

This document describes the expanded RecallForge benchmark corpus and what additional media files should be added manually.

## Summary

- **Text documents**: 54 files (15 original + 39 new)
- **Images**: 10 files (existing)
- **Videos**: 5 compact episodic `.mp4` fixtures plus 5 rich transcript JSON sidecars
- **Documents**: 8 generated `.docx` / `.pptx` / `.pdf` files
- **Total corpus documents**: 82 registered in `CORPUS_DOCS`
- **Total indexed benchmark items**: 82 searchable top-level/sidecar items, plus derived video frame and transcript child memories during video ingest
- **Total benchmark queries**: 231 queries across all modalities

## Episodic Video Corpus

REC-153 replaced the earlier tiny toy clips with a license-safe episodic fixture set. The files are still small enough to commit, but each video now resembles a real personal or work memory: a screen recording, a field clip, a walkthrough, a kitchen note, or a product-planning meeting.

| File | Memory Scenario | Primary Signals |
|------|-----------------|-----------------|
| `coding_demo.mp4` | RecallForge debugging screen recording | code editor, architecture board, action notes, reranking and transcript discussion |
| `nature_timelapse.mp4` | Weekend trail scouting phone clip | forest, mountain, coast, route planning, park/climate notes |
| `architecture_walkthrough.mp4` | Office and system architecture walkthrough | floor plan, service diagram, model diagram, milestone narration |
| `cooking_tutorial.mp4` | Weeknight family recipe memory | pasta, recipe substitutions, handwritten cooking notes |
| `whiteboard_session.mp4` | Product planning meeting | brainstorm board, parent/child memory rollups, benchmark scoring, release actions |

Each `.transcript.json` sidecar now includes:

- timed transcript segments used by video ingest
- a top-level `text` field so the sidecar can also be indexed as a searchable transcript artifact
- `scenario`, `notes`, `related_images`, and `related_documents` metadata for benchmark and documentation provenance

The design follows the same broad shape as episodic-memory video benchmarks such as [Ego4D Episodic Memory](https://ego4d-data.org/docs/benchmarks/episodic-memory/): queries should be able to recover an event, scene, moment, transcript detail, or related artifact from a video-backed memory.

## New Text Documents Added (39 files)

### Technology (5 files)
- `tech_quantum_computing.md` - Quantum computing fundamentals
- `tech_cybersecurity.md` - Cybersecurity best practices
- `tech_blockchain.md` - Blockchain and smart contracts
- `tech_cloud_computing.md` - Cloud computing architecture
- `tech_edge_ai.md` - Edge AI and on-device ML

### Science (5 files)
- `science_genetics.md` - Genetics and DNA sequencing
- `science_climate.md` - Climate change and global warming
- `science_astronomy.md` - Astronomy and deep space
- `science_neuroscience.md` - Neuroscience and brain function
- `science_renewable_energy.md` - Renewable energy technologies

### Cooking (5 files)
- `cooking_baking.md` - Bread baking techniques
- `cooking_asian_cuisine.md` - Asian cooking techniques
- `cooking_bbq.md` - Barbecue and smoking meat
- `cooking_desserts.md` - French pastry and desserts
- `cooking_spices.md` - Understanding spices and flavor profiles

### Sports (5 files)
- `sports_tennis.md` - Tennis strategy and technique
- `sports_swimming.md` - Competitive swimming techniques
- `sports_cycling.md` - Road cycling and racing
- `sports_golf.md` - Golf fundamentals and course management
- `sports_yoga.md` - Yoga philosophy and practice

### History (5 files)
- `history_ancient_rome.md` - Ancient Roman civilization
- `history_renaissance.md` - The Renaissance period
- `history_industrial_revolution.md` - The Industrial Revolution
- `history_ww2.md` - World War II overview
- `history_space_race.md` - The Space Race

### Medicine (5 files)
- `medicine_cardiology.md` - Cardiology and heart health
- `medicine_nutrition.md` - Clinical nutrition and dietetics
- `medicine_mental_health.md` - Mental health and psychiatry
- `medicine_immunology.md` - Immunology and immune system
- `medicine_surgery.md` - Modern surgical techniques

### Music (3 files)
- `music_classical.md` - Classical music history
- `music_jazz.md` - Jazz music and improvisation
- `music_production.md` - Music production and audio engineering

### Art (2 files)
- `art_impressionism.md` - Impressionism art movement
- `art_photography.md` - Photography as fine art

### Finance (2 files)
- `finance_investing.md` - Investment strategies
- `finance_cryptocurrency.md` - Cryptocurrency and digital assets

### Travel (2 files)
- `travel_japan.md` - Traveling in Japan
- `travel_national_parks.md` - US National Parks guide

## Recommended Additional Media Files

To further expand the corpus for more comprehensive cross-modal testing, the following media files should be added manually:

### Recommended Images to Add

1. **tech_quantum_processor.jpg** - Quantum computer processor/chip
2. **tech_cybersecurity_lock.jpg** - Cybersecurity concept with lock/shield
3. **tech_blockchain_network.jpg** - Blockchain network visualization
4. **tech_cloud_servers.jpg** - Cloud data center/server racks
5. **tech_edge_device.jpg** - Edge AI device (smartphone, IoT)
6. **science_dna_helix.jpg** - DNA double helix visualization
7. **science_climate_ice.jpg** - Melting glacier/climate change
8. **science_telescope.jpg** - Observatory/telescope
9. **science_brain_scan.jpg** - Brain MRI/fMRI scan
10. **science_solar_panels.jpg** - Solar panel installation
11. **cooking_bread_loaf.jpg** - Freshly baked bread loaf
12. **cooking_wok.jpg** - Wok with stir-fry ingredients
13. **cooking_smoker.jpg** - BBQ smoker with meat
14. **cooking_pastry.jpg** - French pastries/croissants
15. **cooking_spices.jpg** - Various spices in bowls
16. **sports_tennis_court.jpg** - Tennis court with player
17. **sports_swimmer.jpg** - Swimmer in pool
18. **sports_cyclist.jpg** - Road cyclist
19. **sports_golf_course.jpg** - Golf course green
20. **sports_yoga_pose.jpg** - Yoga pose/position
21. **history_colosseum.jpg** - Roman Colosseum
22. **history_renaissance_painting.jpg** - Renaissance artwork
23. **history_factory.jpg** - Industrial factory scene
24. **history_ww2_photo.jpg** - WWII historical photo
25. **history_rocket_launch.jpg** - Space rocket launch
26. **medicine_heart_diagram.jpg** - Heart anatomy diagram
27. **medicine_healthy_food.jpg** - Healthy nutrition/plate
28. **medicine_therapy_session.jpg` - Mental health therapy
29. **medicine_vaccine.jpg** - Vaccine/antibodies concept
30. **medicine_surgery_room.jpg** - Operating room
31. **music_orchestra.jpg** - Symphony orchestra
32. **music_jazz_band.jpg` - Jazz band performance
33. **music_recording_studio.jpg** - Recording studio
34. **art_monet.jpg** - Impressionist painting (Monet-style)
35. **art_camera.jpg** - Vintage/professional camera
36. **finance_stock_chart.jpg** - Stock market chart
37. **finance_bitcoin.jpg** - Bitcoin/cryptocurrency
38. **travel_tokyo.jpg** - Tokyo cityscape
39. **travel_yosemite.jpg** - Yosemite National Park
40. **travel_grand_canyon.jpg** - Grand Canyon landscape

### Future Real-World Video Additions

The committed corpus is intentionally compact and license-safe. Future benchmark expansions should add opt-in downloaded fixtures or locally supplied clips in these shapes:

1. **meeting_recording_with_slides.mp4** - transcript-heavy meeting with visible slide/document references
2. **screen_recording_debug_trace.mp4** - developer workflow with code, terminal output, and spoken issue context
3. **mobile_walkthrough_errand.mp4** - personal memory clip with objects, location changes, and follow-up tasks
4. **cooking_or_repair_procedure.mp4** - procedural video with step ordering and recipe/tool notes
5. **document_review_session.mp4** - video that references PDFs, decks, and handwritten annotations
6. **travel_or_field_visit_clip.mp4** - visually rich outdoor clip with route, weather, and place notes
7. **classroom_or_tutorial_clip.mp4** - instructional video with transcript-heavy concepts and whiteboard imagery

## Benchmark Query Distribution

| Category | Count | Easy | Medium | Hard | Negative Controls |
|----------|-------|------|--------|------|-------------------|
| text_to_text | 60 | 26 | 24 | 10 | 5 |
| text_to_image | 18 | 8 | 8 | 2 | 0 |
| image_to_text | 20 | 6 | 9 | 5 | 0 |
| image_to_image | 3 | 1 | 2 | 0 | 1 |
| image_to_video | 2 | 0 | 0 | 2 | 0 |
| image_to_document | 20 | 0 | 0 | 20 | 0 |
| video_to_text | 20 | 0 | 12 | 8 | 0 |
| video_to_image | 20 | 0 | 9 | 11 | 0 |
| video_to_video | 1 | 1 | 0 | 0 | 1 |
| video_to_document | 20 | 0 | 0 | 20 | 0 |
| mixed_modal | 20 | 8 | 8 | 4 | 0 |
| text_to_video | 15 | 6 | 6 | 3 | 0 |
| text_to_document | 12 | 6 | 5 | 1 | 0 |
| **Total** | **231** | **62** | **83** | **86** | **7** |

REC-160 expanded the weakest media-query categories to 20 queries each:

- `image_to_text`
- `image_to_document`
- `video_to_text`
- `video_to_image`
- `video_to_document`

Those added queries reuse the existing UAT media corpus with more grounded intent labels and explicit source-path provenance. The corpus should still grow real additional images/videos before treating these as a broad public benchmark.

## New Metrics Added

1. **Precision@K (P@5, P@10)** - Measures precision at rank K with graded relevance
2. **Graded Relevance** - Supports 0/1/2 scores (irrelevant/partial/full)
3. **Query Difficulty Annotation** - Each query tagged as easy/medium/hard
4. **Per-Difficulty Metrics** - Recall reported separately for each difficulty tier
5. **Parent-Memory vs Asset-Level Metrics** - Child assets such as video frames, transcripts, document sections, and OCR pages can credit their canonical parent memory while raw asset metrics remain visible for diagnosis

## Hard Negatives

5 queries designed to test discrimination between similar but distinct topics:
- Ramen noodles (vs pasta)
- Biological neurons (vs AI neural networks)
- Software architecture (vs building architecture)
- Traditional banking (vs cryptocurrency)
- Competitive swimming (vs yoga)

## Negative Controls

5 queries with zero expected results to test precision:
- Underwater basket weaving
- Medieval jousting
- Antique clock repair
- Wild mushroom foraging
- Vintage typewriter maintenance
