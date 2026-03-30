import yaml
from pathlib import Path

class ConfigNode:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            setattr(self, key, value)

class Config:
    def __init__(self, path):
        path = Path(path)
        cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            setattr(self, key, value)


# usage:
# from src.utils.config import Config
# config = Config("config.yml")
