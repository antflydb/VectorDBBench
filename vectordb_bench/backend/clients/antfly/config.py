from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class AntflyConfig(DBConfig):
    host: str = "localhost"
    port: int = 8080
    store_host: str | None = None
    store_port: int | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None
    num_shards: int = 1
    use_direct_store_search: bool = False
    pack_query_vectors: bool = False

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "store_host": self.store_host,
            "store_port": self.store_port,
            "username": self.username.get_secret_value() if self.username else None,
            "password": self.password.get_secret_value() if self.password else None,
            "num_shards": self.num_shards,
            "use_direct_store_search": self.use_direct_store_search,
            "pack_query_vectors": self.pack_query_vectors,
        }


class AntflyIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    num_shards: int = 1
    search_effort: float | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type in (MetricType.IP, MetricType.DP):
            return "inner_product"
        return "l2_squared"

    def index_param(self) -> dict:
        return {"distance_metric": self.parse_metric()}

    def search_param(self) -> dict:
        if self.search_effort is not None:
            return {"search_effort": self.search_effort}
        return {}
