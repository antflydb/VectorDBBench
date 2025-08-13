from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

class CreateTableRequest(BaseModel):
    shards: int
    table_schema: Optional[Dict[str, Any]] = Field(None, alias="tableSchema")

class Table(BaseModel):
    name: str
    shards: int

class AddIndexRequest(BaseModel):
    template: str
    mem_only: bool = Field(..., alias="memOnly")
    dimension: int
    plugin_config: Dict[str, Any] = Field(..., alias="pluginConfig")

class IndexConfig(BaseModel):
    name: str
    status: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    table: str
    full_text_search: Optional[str] = None
    semantic_search: Optional[str] = None
    indexes: Optional[List[str]] = None
    filter_prefix: Optional[str] = None
    embeddings: Optional[Dict[str, List[float]]] = None
    fields: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[Dict[str, bool]] = None
    count: Optional[bool] = None

class QueryResult(BaseModel):
    hits: List[Any]

class Vector(BaseModel):
    id: str
    values: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchRequest(BaseModel):
    inserts: Optional[Dict[str, Any]] = None
    deletes: Optional[List[str]] = None

class UpsertResult(BaseModel):
    upserted_count: int
    response: Any

class DeleteResult(BaseModel):
    deleted_count: int
    response: Any
