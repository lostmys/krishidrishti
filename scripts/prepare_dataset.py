"""Prepare deterministic, duplicate-safe Tomato PlantVillage dataset splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from krishidrishti_ai.config import load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOMATO_CLASS_PREFIX = "Tomato___"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ValidImage:
    path: Path
    sha256: str
    group_id: str | None = None


def resolve_image_root(dataset_dir: Path) -> Path:
    """Support both an official repository root and a directory of class folders."""
    official_color_root = dataset_dir / "raw" / "color"
    return official_color_root if official_color_root.is_dir() else dataset_dir


def discover_tomato_classes(dataset_dir: Path) -> list[Path]:
    image_root = resolve_image_root(dataset_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(f"PlantVillage dataset directory does not exist: {dataset_dir}")
    return sorted(path for path in image_root.iterdir() if path.is_dir() and path.name.startswith(TOMATO_CLASS_PREFIX))


def find_leaf_grouping_metadata(dataset_dir: Path, image_root: Path) -> Path | None:
    """Locate official PlantVillage leaf_grouping/leaf-map.json without guessing contents."""
    candidates = [dataset_dir / "leaf_grouping" / "leaf-map.json"]
    candidates.extend(parent / "leaf_grouping" / "leaf-map.json" for parent in list(image_root.parents)[:3])
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_readable_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image: image.verify()
        with Image.open(path) as image: image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, str(exc)
    return True, None


def validate_and_deduplicate(class_dirs: Iterable[Path]) -> tuple[dict[str, list[ValidImage]], list[dict], list[dict]]:
    by_class: dict[str, list[ValidImage]] = {}; corrupt: list[dict] = []; duplicates: list[dict] = []; first_by_hash: dict[str, Path] = {}
    for class_dir in sorted(class_dirs):
        valid: list[ValidImage] = []
        for path in sorted(item for item in class_dir.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
            readable, error = is_readable_image(path)
            if not readable:
                corrupt.append({"path": str(path), "reason": error}); continue
            digest = sha256_file(path); original = first_by_hash.get(digest)
            if original is not None:
                duplicates.append({"path": str(path), "duplicate_of": str(original), "sha256": digest}); continue
            first_by_hash[digest] = path; valid.append(ValidImage(path, digest))
        by_class[class_dir.name] = valid
    return by_class, corrupt, duplicates


def load_leaf_groups(metadata_path: Path, valid_images: dict[str, list[ValidImage]]) -> tuple[dict[str, list[ValidImage]], int]:
    """Attach official class-qualified leaf IDs; unmapped files become safe singleton groups."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict): raise ValueError(f"Leaf grouping metadata must be a JSON object: {metadata_path}")
    mapped_count = 0; grouped: dict[str, list[ValidImage]] = {}
    for class_name, images in valid_images.items():
        updated: list[ValidImage] = []
        for image in images:
            stem = image.path.stem.lower().strip()
            leaf_key = stem.split("___", 1)[1].strip() if "___" in stem else stem
            options = metadata.get(leaf_key, [])
            matches = [value for value in options if isinstance(value, str) and value.startswith(f"{class_name}:::")]
            group_id = matches[0] if matches else f"{class_name}:::unmapped::{image.sha256}"
            mapped_count += bool(matches); updated.append(ValidImage(image.path, image.sha256, group_id))
        grouped[class_name] = updated
    return grouped, mapped_count


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 3: raise ValueError("Each Tomato class requires at least 3 unique readable images.")
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}; counts = {name: int(raw[name]) for name in SPLIT_NAMES}
    for name in sorted(SPLIT_NAMES, key=lambda item: (raw[item] - counts[item], item), reverse=True)[: total - sum(counts.values())]: counts[name] += 1
    for name in SPLIT_NAMES:
        if counts[name] == 0:
            donor = max(SPLIT_NAMES, key=lambda item: counts[item]); counts[donor] -= 1; counts[name] += 1
    return counts


def create_splits(valid_images: dict[str, list[ValidImage]], ratios: dict[str, float], seed: int) -> dict[str, dict[str, list[ValidImage]]]:
    """Deterministic image-level fallback for datasets with no grouping metadata."""
    if abs(sum(ratios[name] for name in SPLIT_NAMES) - 1.0) > 1e-8: raise ValueError("data.split_ratios must total 1.0")
    result = {name: {} for name in SPLIT_NAMES}
    for class_name, images in sorted(valid_images.items()):
        shuffled = sorted(images, key=lambda image: str(image.path)); random.Random(f"{seed}:{class_name}").shuffle(shuffled); counts = split_counts(len(shuffled), ratios); start = 0
        for split in SPLIT_NAMES:
            end = start + counts[split]; result[split][class_name] = shuffled[start:end]; start = end
    return result


def create_group_aware_splits(grouped_images: dict[str, list[ValidImage]], ratios: dict[str, float], seed: int) -> dict[str, dict[str, list[ValidImage]]]:
    """Assign complete leaf groups to splits; image ratios are targets, never constraints."""
    if abs(sum(ratios[name] for name in SPLIT_NAMES) - 1.0) > 1e-8: raise ValueError("data.split_ratios must total 1.0")
    result = {name: {} for name in SPLIT_NAMES}
    for class_name, images in sorted(grouped_images.items()):
        groups: dict[str, list[ValidImage]] = defaultdict(list)
        for image in images:
            if image.group_id is None: raise ValueError("Group-aware splitting requires a group ID for every image")
            groups[image.group_id].append(image)
        if len(groups) < 3: raise ValueError(f"Class '{class_name}' has only {len(groups)} leaf groups; cannot create three leakage-safe splits.")
        ordered = sorted(groups.items()); random.Random(f"{seed}:{class_name}:groups").shuffle(ordered)
        ordered.sort(key=lambda item: len(item[1]), reverse=True)
        targets = {split: len(images) * ratios[split] for split in SPLIT_NAMES}; assigned = {split: 0 for split in SPLIT_NAMES}
        split_groups: dict[str, list[ValidImage]] = {split: [] for split in SPLIT_NAMES}
        for index, (_, members) in enumerate(ordered):
            if index < 3: split = SPLIT_NAMES[index]
            else: split = max(SPLIT_NAMES, key=lambda name: ((targets[name] - assigned[name]) / max(targets[name], 1.0), name))
            split_groups[split].extend(members); assigned[split] += len(members)
        for split in SPLIT_NAMES: result[split][class_name] = split_groups[split]
    return result


def confirm_no_group_crosses_splits(splits: dict[str, dict[str, list[ValidImage]]]) -> bool:
    observed: dict[str, str] = {}
    for split, classes in splits.items():
        for images in classes.values():
            for image in images:
                if image.group_id is None: continue
                if image.group_id in observed and observed[image.group_id] != split: return False
                observed[image.group_id] = split
    return True


def write_processed_splits(splits: dict[str, dict[str, list[ValidImage]]], destination: Path, overwrite: bool) -> dict[str, dict[str, int]]:
    if destination.exists() and any(destination.iterdir()):
        if not overwrite: raise FileExistsError(f"{destination} is not empty. Re-run with --overwrite to replace generated splits.")
        shutil.rmtree(destination)
    counts = {split: {} for split in SPLIT_NAMES}
    for split, classes in splits.items():
        for class_name, images in classes.items():
            target = destination / split / class_name; target.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(images, start=1): shutil.copy2(image.path, target / f"{index:06d}_{image.path.name}")
            counts[split][class_name] = len(images)
    return counts


def group_statistics(grouped_images: dict[str, list[ValidImage]], splits: dict[str, dict[str, list[ValidImage]]]) -> dict:
    group_sizes: dict[str, int] = defaultdict(int); split_groups = {split: set() for split in SPLIT_NAMES}
    for class_name, images in grouped_images.items():
        for image in images: group_sizes[image.group_id or f"{class_name}:{image.sha256}"] += 1
    for split, classes in splits.items():
        for images in classes.values(): split_groups[split].update(image.group_id for image in images if image.group_id is not None)
    values = list(group_sizes.values())
    return {"group_count": len(group_sizes), "images_per_group": {"min": min(values), "max": max(values), "mean": round(statistics.mean(values), 4), "median": statistics.median(values)}, "split_group_counts": {split: len(groups) for split, groups in split_groups.items()}, "no_group_crosses_splits": confirm_no_group_crosses_splits(splits)}


def write_reports(report_dir: Path, dataset_dir: Path, valid_images: dict[str, list[ValidImage]], corrupt: list[dict], duplicates: list[dict], split_summary: dict[str, dict[str, int]], seed: int, ratios: dict[str, float], splitting: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True); classes = sorted(valid_images)
    mapping = {"dataset": "PlantVillage Tomato baseline", "class_to_index": {name: index for index, name in enumerate(classes)}, "index_to_class": {str(index): name for index, name in enumerate(classes)}}
    total = sum(len(images) for images in valid_images.values()); image_counts = {split: sum(split_summary[split].values()) for split in SPLIT_NAMES}
    summary = {"dataset_dir": str(dataset_dir), "class_prefix": TOMATO_CLASS_PREFIX, "random_seed": seed, "requested_split_ratios": ratios, "actual_split_ratios": {split: image_counts[split] / total for split in SPLIT_NAMES}, "splitting": splitting, "classes": {name: {"unique_readable_images": len(valid_images[name]), "splits": {split: split_summary[split][name] for split in SPLIT_NAMES}} for name in classes}, "totals": {"unique_readable_images": total, "corrupt_files": len(corrupt), "duplicate_files_excluded": len(duplicates), "split_image_counts": image_counts}, "corrupt_files": corrupt, "duplicate_files": duplicates}
    (report_dir / "class_mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_dataset(dataset_dir: Path, destination: Path, report_dir: Path, ratios: dict[str, float], seed: int, overwrite: bool = False) -> dict:
    image_root = resolve_image_root(dataset_dir); classes = discover_tomato_classes(dataset_dir)
    if not classes: raise ValueError(f"No '{TOMATO_CLASS_PREFIX}*' class folders found in {image_root}")
    valid_images, corrupt, duplicates = validate_and_deduplicate(classes); metadata_path = find_leaf_grouping_metadata(dataset_dir, image_root)
    if metadata_path:
        valid_images, mapped = load_leaf_groups(metadata_path, valid_images); splits = create_group_aware_splits(valid_images, ratios, seed)
        splitting = {"mode": "group_aware", "metadata_path": str(metadata_path), "metadata_available": True, "images_matched_to_metadata": mapped, "images_without_metadata_group": sum(len(items) for items in valid_images.values()) - mapped, **group_statistics(valid_images, splits)}
    else:
        splits = create_splits(valid_images, ratios, seed)
        splitting = {"mode": "image_level_fallback", "metadata_available": False, "reason": "No PlantVillage leaf_grouping/leaf-map.json was found.", "no_group_crosses_splits": None}
    split_summary = write_processed_splits(splits, destination, overwrite); write_reports(report_dir, dataset_dir, valid_images, corrupt, duplicates, split_summary, seed, ratios, splitting)
    image_counts = {split: sum(split_summary[split].values()) for split in SPLIT_NAMES}
    return {"classes": len(classes), "splitting_mode": splitting["mode"], "corrupt_files": len(corrupt), "duplicates_excluded": len(duplicates), "split_counts": image_counts, "split_image_counts": image_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Tomato PlantVillage splits without altering image pixels."); parser.add_argument("--config", default="configs/default.yaml"); parser.add_argument("--dataset-dir", help="PlantVillage repository root or a directory containing Tomato___* folders."); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    config = load_config(args.config); dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else Path(config["paths"]["plantvillage_dir"])
    try: outcome = prepare_dataset(dataset_dir, Path(config["paths"]["processed_data_dir"]), Path(config["paths"]["reports_dir"]), config["data"]["split_ratios"], config["data"]["seed"], args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc: raise SystemExit(str(exc)) from exc
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__": main()
