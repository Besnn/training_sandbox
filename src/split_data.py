#!/usr/bin/env python3
"""Split a YOLO dataset while keeping each image paired with its label.

The source dataset must have:
    dataset/
      images/
      labels/
      data.yaml

The output dataset will have:
    split_dataset/
      train/images, train/labels
      val/images, val/labels
      test/images, test/labels
      data.yaml

Unlike splitfolders, this script chooses the split once per image stem and then
copies the matching .txt label to the same split.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("datasets/obb_dataset")
DEFAULT_OUTPUT = Path("datasets/split_obb_dataset")
DEFAULT_RATIO = (0.8, 0.2)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None


def parse_ratio(raw: str) -> tuple[float, float, float]:
    parts = [float(p.strip()) for p in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratio must have three comma-separated values")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("ratio sum must be positive")
    return tuple(p / total for p in parts)  # type: ignore[return-value]


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def collect_samples(input_dir: Path) -> tuple[list[Sample], list[Path]]:
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = image_files(images_dir)
    image_stems = {p.stem for p in images}
    labels_by_stem = {p.stem: p for p in labels_dir.glob("*.txt")}
    samples = [
        Sample(image=img, label=labels_by_stem.get(img.stem))
        for img in images
    ]
    orphan_labels = [
        label for stem, label in labels_by_stem.items()
        if stem not in image_stems
    ]
    return samples, sorted(orphan_labels)


def split_samples(samples: list[Sample], ratio: tuple[float, float, float],
                  seed: int) -> dict[str, list[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def prepare_output(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    for split in SPLITS:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def copy_split(split_name: str, samples: list[Sample], output_dir: Path,
               move: bool, create_empty_labels: bool) -> tuple[int, int]:
    missing_labels = 0
    empty_created = 0
    for sample in samples:
        image_dst = output_dir / split_name / "images" / sample.image.name
        label_dst = output_dir / split_name / "labels" / f"{sample.image.stem}.txt"
        transfer(sample.image, image_dst, move)
        if sample.label is not None:
            transfer(sample.label, label_dst, move)
        else:
            missing_labels += 1
            if create_empty_labels:
                label_dst.write_text("")
                empty_created += 1
    return missing_labels, empty_created


def source_data_yaml(input_dir: Path) -> str | None:
    path = input_dir / "data.yaml"
    return path.read_text() if path.is_file() else None


def write_data_yaml(output_dir: Path, input_dir: Path) -> None:
    src = source_data_yaml(input_dir)
    train = str((output_dir / "train").resolve())
    val = str((output_dir / "val").resolve())
    # test = str((output_dir / "test").resolve())

    if src:
        text = re.sub(r"(?m)^train:\s*.*$", f"train: {train}", src)
        text = re.sub(r"(?m)^val:\s*.*$", f"val: {val}", text)
        # if re.search(r"(?m)^test:\s*", text):
        #     text = re.sub(r"(?m)^test:\s*.*$", f"test: {test}", text)
        # else:
        #     text = f"{text.rstrip()}\ntest: {test}\n"
        # if not re.search(r"(?m)^train:\s*", text):
        #     text = f"train: {train}\nval: {val}\ntest: {test}\n{text}"
    else:
        text = (
            f"train: {train}\n"
            f"val: {val}\n"
            # f"test: {test}\n\n"
            "nc: 4\n"
            'names: ["railroad-crossing", "lights-on", "lights-off", "trefolo"]\n'
            "obb: True\n"
        )
    (output_dir / "data.yaml").write_text(text.rstrip() + "\n")


def print_summary(splits: dict[str, list[Sample]], missing_by_split: dict[str, int],
                  empty_by_split: dict[str, int], orphan_labels: list[Path]) -> None:
    print("\nSplit complete.")
    for split in SPLITS:
        samples = splits[split]
        annotated = sum(1 for sample in samples if sample.label is not None)
        print(
            f"  {split:<5} images={len(samples):>5} "
            f"labels={annotated:>5} "
            f"missing_labels={missing_by_split[split]:>5} "
            f"empty_created={empty_by_split[split]:>5}"
        )
    if orphan_labels:
        print(f"\n[WARN] {len(orphan_labels)} label files had no matching image and were not copied.")
        for label in orphan_labels[:20]:
            print(f"       {label}")
        if len(orphan_labels) > 20:
            print(f"       ... {len(orphan_labels) - 20} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Source YOLO dataset directory containing images/ and labels/.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output split dataset directory.")
    parser.add_argument("--ratio", type=parse_ratio, default=DEFAULT_RATIO,
                        help="Train,val,test ratio, e.g. 0.8,0.1,0.1.")
    parser.add_argument("--seed", type=int, default=23,
                        help="Random seed for deterministic splits.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying them.")
    parser.add_argument("--no-clean", action="store_true",
                        help="Do not delete the output directory before splitting.")
    parser.add_argument("--no-empty-labels", action="store_true",
                        help="Do not create empty .txt files for images with no source label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    samples, orphan_labels = collect_samples(input_dir)
    if not samples:
        raise SystemExit(f"No images found in {input_dir / 'images'}")

    splits = split_samples(samples, args.ratio, args.seed)
    prepare_output(output_dir, clean=not args.no_clean)

    missing_by_split = {}
    empty_by_split = {}
    for split, split_samples_ in splits.items():
        missing, empty_created = copy_split(
            split,
            split_samples_,
            output_dir,
            move=args.move,
            create_empty_labels=not args.no_empty_labels,
        )
        missing_by_split[split] = missing
        empty_by_split[split] = empty_created

    write_data_yaml(output_dir, input_dir)
    print_summary(splits, missing_by_split, empty_by_split, orphan_labels)
    print(f"\nData YAML written to: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
