from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class AntflyConfig(DBConfig):
    url: SecretStr

    def to_dict(self) -> dict:
        # The antfly-py client expects host, port, and scheme separately.
        # We'll parse the URL to provide these.
        from urllib.parse import urlparse
        parsed_url = urlparse(self.url.get_secret_value())
        return {
            "host": parsed_url.hostname,
            "port": parsed_url.port,
            "scheme": parsed_url.scheme,
        }


class AntflyIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    dimension: int
    template: str = "string"
    mem_only: bool = False
    provider: str = "ollama"
    model: str | None = None
    embedding_url: str | None = None

    def index_param(self) -> dict:
        params = {
            "dimension": self.dimension,
            "template": self.template,
            "mem_only": self.mem_only,
            "pluginConfig": {
                "provider": self.provider,
            }
        }
        if self.model:
            params["pluginConfig"]["model"] = self.model
        if self.embedding_url:
            params["pluginConfig"]["url"] = self.embedding_url
        return params

    def search_param(self) -> dict:
        return {}
