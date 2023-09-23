from pathlib import Path

from confz import BaseConfig, FileSource

CONFIG_DIR = Path(__file__).parent.resolve() / "config"


class Config(BaseConfig):
    matches_path: str
    outcomes_paths: dict
    ratings_paths: dict

    CONFIG_SOURCES = FileSource(file=CONFIG_DIR / "params.yml")
