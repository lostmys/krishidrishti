from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prepare_dataset import (  # noqa: E402
    ValidImage,
    confirm_no_group_crosses_splits,
    create_group_aware_splits,
    create_splits,
    discover_tomato_classes,
    prepare_dataset,
    validate_and_deduplicate,
)


RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_discovers_only_plantvillage_tomato_classes(tmp_path: Path) -> None:
    (tmp_path / "Tomato___healthy").mkdir()
    (tmp_path / "Tomato___Early_blight").mkdir()
    (tmp_path / "Potato___healthy").mkdir()
    assert [path.name for path in discover_tomato_classes(tmp_path)] == ["Tomato___Early_blight", "Tomato___healthy"]


def test_split_is_deterministic_and_preserves_70_15_15_counts() -> None:
    images = [ValidImage(Path(f"image_{index:03d}.jpg"), str(index)) for index in range(20)]
    first = create_splits({"Tomato___healthy": images}, RATIOS, seed=42)
    second = create_splits({"Tomato___healthy": images}, RATIOS, seed=42)
    assert {name: len(first[name]["Tomato___healthy"]) for name in first} == {"train": 14, "val": 3, "test": 3}
    assert first == second


def test_duplicate_detection_excludes_duplicate_and_reports_corrupt_file(tmp_path: Path) -> None:
    class_dir = tmp_path / "Tomato___healthy"
    write_image(class_dir / "source.png", (10, 20, 30))
    (class_dir / "duplicate.png").write_bytes((class_dir / "source.png").read_bytes())
    (class_dir / "broken.jpg").write_bytes(b"not an image")
    valid, corrupt, duplicates = validate_and_deduplicate([class_dir])
    assert len(valid["Tomato___healthy"]) == 1
    assert len(corrupt) == 1
    assert len(duplicates) == 1
    assert duplicates[0]["duplicate_of"].endswith("duplicate.png")
    assert duplicates[0]["path"].endswith("source.png")


def test_prepare_writes_splits_and_reports(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "PlantVillage"
    for class_index, class_name in enumerate(("Tomato___Bacterial_spot", "Tomato___healthy")):
        for image_index in range(20):
            write_image(dataset_dir / class_name / f"{image_index}.png", (class_index, image_index, 100))
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    outcome = prepare_dataset(dataset_dir, processed, reports, RATIOS, seed=7)
    assert outcome["split_counts"] == {"train": 28, "val": 6, "test": 6}
    mapping = json.loads((reports / "class_mapping.json").read_text(encoding="utf-8"))
    summary = json.loads((reports / "dataset_summary.json").read_text(encoding="utf-8"))
    assert mapping["class_to_index"]["Tomato___healthy"] == 1
    assert summary["totals"]["unique_readable_images"] == 40
    assert summary["splitting"]["mode"] == "image_level_fallback"
    assert len(list((processed / "train" / "Tomato___healthy").glob("*"))) == 14


def test_group_aware_split_never_leaks_leaf_groups_and_is_deterministic() -> None:
    images = []
    for group_index in range(6):
        for image_index in range(2):
            images.append(ValidImage(Path(f"leaf_{group_index}_{image_index}.jpg"), f"{group_index}-{image_index}", f"Tomato___healthy:::{group_index}"))
    first = create_group_aware_splits({"Tomato___healthy": images}, RATIOS, seed=11)
    second = create_group_aware_splits({"Tomato___healthy": images}, RATIOS, seed=11)
    assert first == second
    assert confirm_no_group_crosses_splits(first)
    assignments = {image.group_id: split for split, classes in first.items() for image in classes["Tomato___healthy"]}
    assert len(assignments) == 6
    for group_id in assignments:
        assert {split for split, classes in first.items() if any(image.group_id == group_id for image in classes["Tomato___healthy"])} == {assignments[group_id]}


def test_official_leaf_map_activates_group_aware_preparation(tmp_path: Path) -> None:
    root = tmp_path / "PlantVillage"
    image_dir = root / "raw" / "color" / "Tomato___healthy"
    leaf_map: dict[str, list[str]] = {}
    for group_index in range(6):
        for image_index in range(2):
            stem = f"leaf_{group_index}_{image_index}"
            write_image(image_dir / f"{stem}.png", (group_index, image_index, 100))
            leaf_map[stem] = [f"Tomato___healthy:::{group_index}"]
    metadata_dir = root / "leaf_grouping"; metadata_dir.mkdir(parents=True)
    (metadata_dir / "leaf-map.json").write_text(json.dumps(leaf_map), encoding="utf-8")
    reports = tmp_path / "reports"
    outcome = prepare_dataset(root, tmp_path / "processed", reports, RATIOS, seed=9)
    summary = json.loads((reports / "dataset_summary.json").read_text(encoding="utf-8"))
    assert outcome["splitting_mode"] == "group_aware"
    assert summary["splitting"]["group_count"] == 6
    assert summary["splitting"]["images_per_group"]["min"] == 2
    assert summary["splitting"]["no_group_crosses_splits"] is True
