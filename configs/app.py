import yaml
from pathlib import Path

def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Basic config structure that could be expanded
class AppConfig:
    def __init__(self):
        self.config_dir = Path(__file__).resolve().parent
        self.embeddings = load_yaml_config(self.config_dir / "embeddings.yaml")
        self.ranker = load_yaml_config(self.config_dir / "ranker.yaml")

config = AppConfig()
