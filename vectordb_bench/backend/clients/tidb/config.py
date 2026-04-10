from typing import Any, TypedDict

from pydantic import BaseModel, SecretStr, model_validator

from ..api import DBCaseConfig, DBConfig, MetricType


class TiDBConfigDict(TypedDict):
    host: str
    port: int
    user: str
    password: str
    database: str
    ssl_verify_cert: bool
    ssl_verify_identity: bool


class TiDBConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 4000
    db_name: str = "test"
    ssl: bool = False

    def to_dict(self) -> TiDBConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
            "database": self.db_name,
            "ssl_verify_cert": self.ssl,
            "ssl_verify_identity": self.ssl,
        }

    @model_validator(mode="before")
    @classmethod
    def not_empty_field(cls, data: Any) -> Any:
        if isinstance(data, dict):
            skip = {"password", "db_label"}
            for name, v in data.items():
                if name in skip:
                    continue
                if isinstance(v, str) and len(v) == 0:
                    msg = f"Empty string for field '{name}'!"
                    raise ValueError(msg)
        return data


class TiDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def get_metric_fn(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vec_l2_distance"
        if self.metric_type == MetricType.COSINE:
            return "vec_cosine_distance"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }

    def search_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }
