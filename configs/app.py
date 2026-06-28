import os
import yaml

def load_yaml(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_config():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = {}
    config.update(load_yaml(os.path.join(root_dir, "config.yaml")))
    config["embeddings"] = load_yaml(os.path.join(root_dir, "configs", "embeddings.yaml"))
    config["ranker"] = load_yaml(os.path.join(root_dir, "configs", "ranker.yaml"))
    config["parser"] = load_yaml(os.path.join(root_dir, "configs", "parser.yaml"))
    config["scoring"] = load_yaml(os.path.join(root_dir, "configs", "scoring.yaml"))
    config["weights"] = load_yaml(os.path.join(root_dir, "configs", "weights.yaml"))
    config["logging"] = load_yaml(os.path.join(root_dir, "configs", "logging.yaml"))
    return config
