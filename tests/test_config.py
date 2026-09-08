from pathlib import Path

from krishidrishti_ai.config import load_config


def test_experiment_config_inherits_default_and_resolves_project_paths() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "experiments" / "efficientnet_b0.yaml")
    assert config["model"]["name"] == "efficientnet_b0"
    assert config["data"]["seed"] == 42
    assert Path(config["paths"]["checkpoint_path"]) == root / "artifacts" / "checkpoints" / "efficientnet_b0.pt"


def test_production_registry_resolves_tomato_robust_v2_and_preserves_cotton_soyabean() -> None:
    root = Path(__file__).parents[1]
    default_cfg = load_config(root / "configs" / "default.yaml")

    crops = default_cfg["crops"]
    assert crops["cotton"] == "configs/experiments/cotton_mobilenetv3.yaml"
    assert crops["soyabean"] == "configs/experiments/soyabean_mobilenetv3.yaml"
    assert crops["tomato"] == "configs/experiments/tomato_mobilenetv3.yaml"

    cotton_cfg = load_config(root / crops["cotton"])
    assert "cotton" in cotton_cfg["paths"]["checkpoint_path"]

    soyabean_cfg = load_config(root / crops["soyabean"])
    assert "soyabean" in soyabean_cfg["paths"]["checkpoint_path"]

    tomato_cfg = load_config(root / crops["tomato"])
    ckpt_path = Path(tomato_cfg["paths"]["checkpoint_path"])
    assert ckpt_path.name == "mobilenetv3_robust_v2.pt"
    assert ckpt_path.is_file()

    cal_path = Path(tomato_cfg["inference"]["calibration_path"])
    assert cal_path.name == "calibration_robust_v2.json"
    assert cal_path.is_file()
    assert tomato_cfg["inference"]["temperature"] == 1.3377

