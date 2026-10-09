"""Create deterministic contact sheets for visual HandSeal dataset review."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .audit_dataset import DEFAULT_CLASSES, IMAGE_EXTENSIONS


def image_paths(root: Path, split: str, label: str) -> list[Path]:
    directory = root / split / label
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_contact_sheet(
    paths: list[Path],
    title: str,
    output: Path,
    sample_limit: int = 25,
    columns: int = 5,
) -> None:
    selected = paths[:sample_limit]
    tile_size = 180
    label_height = 30
    rows = max(1, (len(selected) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "#20242b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, path in enumerate(selected):
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_size - 8, tile_size - 8), Image.Resampling.LANCZOS)
                left = x + (tile_size - image.width) // 2
                top = y + (tile_size - image.height) // 2
                sheet.paste(image, (left, top))
            name = path.stem[:24]
        except Exception:
            name = "CORRUPT"
            draw.rectangle((x + 4, y + 4, x + tile_size - 4, y + tile_size - 4), outline="#ff5050", width=3)
        draw.text((x + 4, y + tile_size + 4), name, fill="white", font=font)

    draw.text((8, 8), f"{title} | samples={len(selected)} total={len(paths)}", fill="#ffd166", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="handseal_dataset")
    parser.add_argument("--output-dir", default="reports/dataset_review")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--samples-per-class", type=int, default=25)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.columns < 1:
        raise SystemExit("--samples-per-class and --columns must be positive.")

    root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir)
    generated = 0
    for split in ("train", "valid", "test"):
        for label in args.classes:
            paths = image_paths(root, split, label)
            output = output_dir / f"{split}_{label}.jpg"
            create_contact_sheet(
                paths,
                f"{split}/{label}",
                output,
                sample_limit=args.samples_per_class,
                columns=args.columns,
            )
            generated += 1
    print(f"Contact sheets: {generated}")
    print(f"Output directory: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
