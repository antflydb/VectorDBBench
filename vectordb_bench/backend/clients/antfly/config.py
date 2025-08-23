from typing import Optional
from ..api import DBConfig

class AntflyConfig(DBConfig):
    host: Optional[str] = "localhost"
    port: Optional[int] = 8080

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
        }
