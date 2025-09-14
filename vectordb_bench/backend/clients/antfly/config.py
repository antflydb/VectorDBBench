from pydantic import SecretStr

from ..api import DBConfig


class AntflyConfig(DBConfig):
    def to_dict(self) -> dict:
        return {}
