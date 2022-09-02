from pathlib import Path

from confz import ConfZ, ConfZFileSource

CONFIG_DIR = Path(__file__).parent.resolve() / "config"


class Config(ConfZ):
    project_path: str
    database: dict

    CONFIG_SOURCES = ConfZFileSource(file=CONFIG_DIR / "params.yml")
