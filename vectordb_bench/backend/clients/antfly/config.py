from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class AntflyConfig(DBConfig):
    host: str = "localhost"
    port: int = 8080
    username: SecretStr | None = None
    password: SecretStr | None = None
    num_shards: int = 1

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username.get_secret_value() if self.username else None,
            "password": self.password.get_secret_value() if self.password else None,
            "num_shards": self.num_shards,
        }


class AntflyIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    num_shards: int = 1

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type in (MetricType.IP, MetricType.DP):
            return "inner_product"
        return "l2_squared"

    def index_param(self) -> dict:
        return {"distance_metric": self.parse_metric()}

    def search_param(self) -> dict:
        return {}
