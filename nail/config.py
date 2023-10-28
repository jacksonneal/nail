"""Configure settings.

Copyright (C) 2023 Jackson Neal
"""
from os.path import join

from pydantic import validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .validator import dir_exists


class Settings(BaseSettings):
    """Settings."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="nail_",
    )

    data_dir: str = "replace-me!"
    data_version: str = "v4.1"
    feature_set: str = "medium"
    models_dir: str = "replace-me!"
    xgb_tree_method: str = "hist"

    @property
    def version_data_dir(self):
        return join(self.data_dir, self.data_version)

    @validator("data_dir", "models_dir", always=True)
    def check_data_dir(cls, v: str) -> str:
        return dir_exists(v)


settings = Settings()
