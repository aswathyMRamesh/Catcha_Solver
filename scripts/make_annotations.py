import csv
from pathlib import Path

# Root of your dataset (update if needed)
dataset_root = Path("~/captcha_solver/data/UTN-CV25-Captcha-Dataset").expanduser()

# Find all "images" folders recursively
for images_dir in dataset_root.rglob("images"):
    csv_path = images_dir.parent / "annotations.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for img in sorted(images_dir.glob("*.png")):  # sort for consistency
            label = img.stem  # filename without extension
            writer.writerow([img.name, label])

    print(f"✅ annotations.csv created at {csv_path}")
