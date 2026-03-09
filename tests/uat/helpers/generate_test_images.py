#!/usr/bin/env python3
"""
Generate synthetic test images for UAT cross-modal testing.

Creates 10 images with distinct visual content and embedded text
so that cross-modal search can meaningfully match them.
"""

import os
import sys
import random
import math

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow is required. pip install pillow")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use default font (available everywhere)
def get_font(size=20):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def draw_whiteboard_diagram(path):
    """Simulate a whiteboard with boxes and arrows - system architecture."""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(16)
    title_font = get_font(24)

    draw.text((250, 20), "System Architecture", fill="black", font=title_font)

    # Draw boxes
    boxes = [
        (50, 100, 200, 170, "Frontend\nReact App"),
        (300, 100, 500, 170, "API Gateway\nNginx"),
        (550, 100, 750, 170, "Auth Service\nOAuth2"),
        (150, 250, 350, 320, "Backend\nPython/FastAPI"),
        (450, 250, 650, 320, "ML Service\nPyTorch"),
        (150, 400, 350, 470, "PostgreSQL\nDatabase"),
        (450, 400, 650, 470, "Redis\nCache"),
    ]
    for x1, y1, x2, y2, label in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
        draw.text((x1 + 10, y1 + 10), label, fill="navy", font=font)

    # Draw arrows
    arrows = [(200, 135, 300, 135), (500, 135, 550, 135),
              (400, 170, 250, 250), (400, 170, 550, 250),
              (250, 320, 250, 400), (550, 320, 550, 400)]
    for x1, y1, x2, y2 in arrows:
        draw.line([x1, y1, x2, y2], fill="gray", width=2)

    img.save(path)


def draw_whiteboard_brainstorm(path):
    """Simulate whiteboard brainstorming session with mind map."""
    img = Image.new("RGB", (800, 600), "#FFFFF0")
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    title_font = get_font(20)

    # Central topic
    draw.ellipse([300, 250, 500, 350], outline="red", width=3)
    draw.text((340, 280), "AI Strategy", fill="red", font=title_font)

    # Branches
    branches = [
        (150, 100, "Data Pipeline"), (600, 100, "Model Training"),
        (100, 400, "Deployment"), (650, 400, "Monitoring"),
        (50, 250, "Team Skills"), (700, 250, "Budget"),
    ]
    for x, y, label in branches:
        draw.ellipse([x-50, y-25, x+50, y+25], outline="blue", width=2)
        draw.text((x-40, y-10), label, fill="blue", font=font)
        draw.line([400, 300, x, y], fill="gray", width=1)

    # Sticky note simulation
    draw.rectangle([10, 520, 200, 590], fill="yellow", outline="orange")
    draw.text((20, 530), "TODO: Review Q3\nbudget proposal", fill="black", font=font)

    img.save(path)


def draw_handwritten_notes(path):
    """Simulate handwritten meeting notes."""
    img = Image.new("RGB", (600, 800), "#FFF8DC")
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    title_font = get_font(24)

    draw.text((50, 30), "Meeting Notes - Project Review", fill="darkblue", font=title_font)
    draw.line([50, 60, 550, 60], fill="lightblue", width=1)

    lines = [
        "Date: March 5, 2026",
        "Attendees: Brian, Sarah, Mike",
        "",
        "1. Sprint Review",
        "   - Completed 8/10 stories",
        "   - Velocity improved 15%",
        "   - Demo went well with client",
        "",
        "2. Technical Debt",
        "   - Need to refactor auth module",
        "   - Database migrations pending",
        "   - Unit test coverage at 72%",
        "",
        "3. Action Items",
        "   [ ] Brian: Review PR #245",
        "   [ ] Sarah: Update API docs",
        "   [ ] Mike: Deploy staging env",
        "",
        "4. Next Sprint Planning",
        "   - Focus on performance",
        "   - Add caching layer",
        "   - Improve error handling",
    ]

    y = 80
    for line in lines:
        # Slightly wobbly positioning to simulate handwriting
        x_offset = random.randint(-2, 2)
        draw.text((60 + x_offset, y), line, fill="darkblue", font=font)
        y += 30
        if y % 90 == 0:
            draw.line([50, y-5, 550, y-5], fill="lightblue", width=1)

    img.save(path)


def draw_architecture_blueprint(path):
    """Simulate a floor plan / architectural blueprint."""
    img = Image.new("RGB", (800, 600), "#E8F0FE")
    draw = ImageDraw.Draw(img)
    font = get_font(12)
    title_font = get_font(20)

    draw.text((250, 10), "Floor Plan - Level 1", fill="navy", font=title_font)

    # Outer walls
    draw.rectangle([50, 50, 750, 550], outline="navy", width=3)

    # Rooms
    rooms = [
        (50, 50, 350, 300, "Living Room\n24' x 18'"),
        (350, 50, 550, 300, "Kitchen\n14' x 18'"),
        (550, 50, 750, 300, "Dining\n14' x 18'"),
        (50, 300, 250, 550, "Bedroom 1\n14' x 18'"),
        (250, 300, 500, 550, "Bedroom 2\n18' x 18'"),
        (500, 300, 750, 550, "Garage\n18' x 18'"),
    ]
    for x1, y1, x2, y2, label in rooms:
        draw.rectangle([x1, y1, x2, y2], outline="navy", width=2)
        cx = (x1 + x2) // 2 - 30
        cy = (y1 + y2) // 2 - 10
        draw.text((cx, cy), label, fill="navy", font=font)

    # Door openings (gaps in walls)
    for x in [180, 420, 640, 150, 370]:
        draw.rectangle([x, 298, x+30, 302], fill="#E8F0FE")

    # Scale bar
    draw.line([50, 580, 150, 580], fill="black", width=2)
    draw.text((70, 582), "10 ft", fill="black", font=font)

    # North arrow
    draw.polygon([(720, 570), (730, 555), (740, 570)], fill="black")
    draw.text((725, 572), "N", fill="black", font=font)

    img.save(path)


def draw_food_photo(path):
    """Simulate a food photo - plate of pasta."""
    img = Image.new("RGB", (600, 600), "#F5E6D3")
    draw = ImageDraw.Draw(img)
    font = get_font(14)

    # Plate circle
    draw.ellipse([100, 100, 500, 500], fill="white", outline="#DDD", width=2)

    # Pasta squiggles (simulated)
    for _ in range(30):
        x = random.randint(180, 420)
        y = random.randint(180, 420)
        dx = random.randint(20, 60)
        dy = random.randint(-20, 20)
        color = random.choice(["#F0C040", "#E8B830", "#D4A020"])
        draw.arc([x, y, x+dx, y+abs(dy)+10], 0, 180, fill=color, width=3)

    # Red sauce
    for _ in range(15):
        x = random.randint(200, 400)
        y = random.randint(200, 400)
        r = random.randint(5, 15)
        draw.ellipse([x-r, y-r, x+r, y+r], fill="#CC3333")

    # Basil leaves
    for _ in range(5):
        x = random.randint(220, 380)
        y = random.randint(220, 380)
        draw.ellipse([x, y, x+20, y+12], fill="#228B22")

    # Parmesan
    for _ in range(20):
        x = random.randint(200, 400)
        y = random.randint(200, 400)
        draw.rectangle([x, y, x+4, y+3], fill="#FFE4B5")

    draw.text((220, 520), "Fresh Pasta with Tomato Basil", fill="#333", font=font)

    img.save(path)


def draw_nature_forest(path):
    """Simulate a forest landscape photo."""
    img = Image.new("RGB", (800, 600), "#87CEEB")
    draw = ImageDraw.Draw(img)

    # Ground
    draw.rectangle([0, 400, 800, 600], fill="#228B22")

    # Trees
    for x in range(50, 800, 80):
        trunk_h = random.randint(100, 200)
        trunk_w = random.randint(15, 25)
        ty = 400 - trunk_h
        draw.rectangle([x-trunk_w//2, ty, x+trunk_w//2, 400], fill="#8B4513")
        # Canopy
        for layer in range(3):
            r = random.randint(30, 60)
            cy = ty - layer * 20
            green = random.choice(["#006400", "#228B22", "#2E8B57", "#3CB371"])
            draw.ellipse([x-r, cy-r, x+r, cy+r//2], fill=green)

    # Sun
    draw.ellipse([650, 30, 730, 110], fill="#FFD700")

    # Path
    draw.polygon([(350, 600), (450, 600), (410, 400), (390, 400)], fill="#D2B48C")

    img.save(path)


def draw_nature_ocean(path):
    """Simulate an ocean/beach scene."""
    img = Image.new("RGB", (800, 600), "#1E90FF")
    draw = ImageDraw.Draw(img)

    # Gradient sky to ocean
    for y in range(300):
        blue = int(135 + (y / 300) * 120)
        draw.line([(0, y), (800, y)], fill=(100, 149, min(blue, 255)))

    # Waves
    for y in range(300, 480, 15):
        for x in range(0, 800, 40):
            offset = random.randint(-5, 5)
            draw.arc([x+offset, y, x+40+offset, y+15], 0, 180,
                     fill="white", width=2)

    # Sand
    draw.rectangle([0, 480, 800, 600], fill="#F4A460")

    # Shells
    for _ in range(10):
        x = random.randint(50, 750)
        y = random.randint(490, 580)
        draw.ellipse([x, y, x+8, y+6], fill="#FFF8DC", outline="#DEB887")

    img.save(path)


def draw_nature_mountain(path):
    """Simulate a mountain landscape."""
    img = Image.new("RGB", (800, 600), "#87CEEB")
    draw = ImageDraw.Draw(img)

    # Mountains
    mountains = [
        [(0, 400), (200, 150), (400, 400)],
        [(150, 400), (400, 100), (650, 400)],
        [(400, 400), (600, 180), (800, 400)],
    ]
    colors = ["#696969", "#808080", "#A9A9A9"]
    for pts, color in zip(mountains, colors):
        draw.polygon(pts, fill=color)
        # Snow caps
        peak = min(pts, key=lambda p: p[1])
        draw.polygon([
            (peak[0]-30, peak[1]+40),
            peak,
            (peak[0]+30, peak[1]+40)
        ], fill="white")

    # Meadow
    draw.rectangle([0, 400, 800, 600], fill="#90EE90")

    # Wildflowers
    for _ in range(30):
        x = random.randint(10, 790)
        y = random.randint(410, 590)
        color = random.choice(["red", "yellow", "purple", "white"])
        draw.ellipse([x-3, y-3, x+3, y+3], fill=color)

    img.save(path)


def draw_technical_diagram(path):
    """Simulate a neural network diagram."""
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(14)
    title_font = get_font(20)

    draw.text((250, 20), "Neural Network Architecture", fill="black", font=title_font)

    layers = [
        ("Input\nLayer", 3, 100),
        ("Hidden\nLayer 1", 5, 250),
        ("Hidden\nLayer 2", 5, 400),
        ("Output\nLayer", 2, 550),
    ]

    node_positions = {}
    for label, count, x in layers:
        draw.text((x-20, 530), label, fill="gray", font=font)
        spacing = 400 // (count + 1)
        for i in range(count):
            y = 80 + spacing * (i + 1)
            node_positions[(x, y)] = True
            draw.ellipse([x-15, y-15, x+15, y+15], fill="#4169E1", outline="navy")

    # Connections between adjacent layers
    for i in range(len(layers) - 1):
        _, c1, x1 = layers[i]
        _, c2, x2 = layers[i+1]
        s1 = 400 // (c1 + 1)
        s2 = 400 // (c2 + 1)
        for j in range(c1):
            y1 = 80 + s1 * (j + 1)
            for k in range(c2):
                y2 = 80 + s2 * (k + 1)
                draw.line([x1+15, y1, x2-15, y2], fill="#B0C4DE", width=1)

    img.save(path)


def draw_screenshot_code(path):
    """Simulate a code editor screenshot."""
    img = Image.new("RGB", (800, 600), "#1E1E1E")
    draw = ImageDraw.Draw(img)
    font = get_font(14)

    # Tab bar
    draw.rectangle([0, 0, 800, 30], fill="#2D2D2D")
    draw.text((10, 5), "search.py", fill="#CCCCCC", font=font)
    draw.text((120, 5), "backend.py", fill="#888888", font=font)

    # Line numbers and code
    code_lines = [
        ("def", " hybrid_search", "(self, query: str) -> List[Result]:"),
        ("    ", '"""Run full hybrid search pipeline."""', ""),
        ("    ", "# Step 1: BM25 probe", ""),
        ("    ", "fts_results = self._bm25_probe(query)", ""),
        ("    ", "", ""),
        ("    ", "# Step 2: Query expansion", ""),
        ("    ", "if", " self.backend.needs_expander():"),
        ("        ", "expansions = self._expand(query)", ""),
        ("    ", "", ""),
        ("    ", "# Step 3: Parallel vector search", ""),
        ("    ", "with", " ThreadPoolExecutor() ", "as", " pool:"),
        ("        ", "futures = {pool.submit(s): s ", "for", " s ", "in", " searches}"),
        ("    ", "", ""),
        ("    ", "# Step 4: RRF fusion", ""),
        ("    ", "candidates = self._rrf_fuse(all_results)", ""),
        ("    ", "", ""),
        ("    ", "# Step 5: Reranking", ""),
        ("    ", "scores = self.backend.rerank(query, candidates)", ""),
        ("    ", "return", " self._blend(candidates, scores)"),
    ]

    y = 40
    for i, parts in enumerate(code_lines, 1):
        draw.text((10, y), str(i).rjust(3), fill="#858585", font=font)
        x = 50
        colors = ["#569CD6", "#DCDCAA", "#CCCCCC", "#C586C0", "#CE9178",
                  "#4EC9B0", "#9CDCFE"]
        for j, part in enumerate(parts):
            color = colors[j % len(colors)]
            if part.startswith("#"):
                color = "#6A9955"
            elif part.startswith('"""'):
                color = "#CE9178"
            draw.text((x, y), part, fill=color, font=font)
            x += len(part) * 8
        y += 25

    img.save(path)


# Generate all images
images = [
    ("whiteboard_architecture.png", draw_whiteboard_diagram),
    ("whiteboard_brainstorm.png", draw_whiteboard_brainstorm),
    ("handwritten_notes.png", draw_handwritten_notes),
    ("floor_plan_blueprint.png", draw_architecture_blueprint),
    ("food_pasta_dish.png", draw_food_photo),
    ("forest_landscape.png", draw_nature_forest),
    ("ocean_beach.png", draw_nature_ocean),
    ("mountain_landscape.png", draw_nature_mountain),
    ("neural_network_diagram.png", draw_technical_diagram),
    ("code_editor_screenshot.png", draw_screenshot_code),
]

if __name__ == "__main__":
    random.seed(42)
    for filename, draw_func in images:
        path = os.path.join(OUTPUT_DIR, filename)
        draw_func(path)
        size = os.path.getsize(path)
        print(f"  Created: {filename} ({size:,} bytes)")
    print(f"\nGenerated {len(images)} test images in {OUTPUT_DIR}")
