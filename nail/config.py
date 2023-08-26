from os.path import join
from typing import Annotated

from pydantic import AfterValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .validator import dir_exists


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="nail_",
    )

    data_dir: Annotated[
        str, AfterValidator(dir_exists), Field(validate_default=True)
    ] = "replace-me"
    data_version: str = "v4.1"
    feature_set: str = "small"

    @property
    def version_data_dir(self):
        return join(self.data_dir, self.data_version)


settings = Settings()
