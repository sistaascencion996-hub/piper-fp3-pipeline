import json
from pathlib import Path

def test_public_config_parses():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "config" / "pipeline.example.json").read_text())
    assert cfg["robot"]["model"] == "Piper"
    assert cfg["training"]["optimizer_steps"] == 10000
    assert cfg["training"]["checkpoint_filename"] == "model_best_train_loss.pth"
    assert cfg["inference"]["replan_every_step"] is False
