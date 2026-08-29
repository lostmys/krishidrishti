from pathlib import Path

from krishidrishti_ai.config import load_config


def test_experiment_config_inherits_default_and_resolves_project_paths() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "experiments" / "efficientnet_b0.yaml")
    assert config["model"]["name"] == "efficientnet_b0"
    assert config["data"]["seed"] == 42
    assert Path(config["paths"]["checkpoint_path"]) == root / "artifacts" / "checkpoints" / "efficientnet_b0.pt"
