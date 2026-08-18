from importlib import import_module
from typing import Any

import pytest

from vectordb_bench.backend.filter import FilterOp, IntFilter, non_filter


@pytest.fixture
def numeric_filter() -> IntFilter:
    return IntFilter(filter_rate=0.99, int_field="id", int_value=49_500)


def _adapter_class(dependency: str, module: str, name: str) -> Any:
    pytest.importorskip(dependency)
    return getattr(import_module(module), name)


def test_qdrant_numeric_filter_is_inclusive(numeric_filter: IntFilter):
    adapter_cls = _adapter_class(
        "qdrant_client",
        "vectordb_bench.backend.clients.qdrant_local.qdrant_local",
        "QdrantLocal",
    )

    assert FilterOp.NumGE in adapter_cls.supported_filter_types
    adapter = adapter_cls.__new__(adapter_cls)
    adapter._primary_field = "pk"
    adapter.prepare_filter(numeric_filter)
    numeric_range = adapter._query_filter.must[0].range
    assert numeric_range.gte == 49_500
    assert numeric_range.gt is None
    adapter.prepare_filter(non_filter)
    assert adapter._query_filter is None


def test_weaviate_numeric_filter_is_inclusive(numeric_filter: IntFilter):
    adapter_cls = _adapter_class(
        "weaviate",
        "vectordb_bench.backend.clients.weaviate_cloud.weaviate_cloud",
        "WeaviateCloud",
    )

    assert FilterOp.NumGE in adapter_cls.supported_filter_types
    adapter = adapter_cls.__new__(adapter_cls)
    adapter._scalar_field = "key"
    adapter.prepare_filter(numeric_filter)
    assert adapter._where_filter == {
        "path": ["key"],
        "operator": "GreaterThanEqual",
        "valueInt": 49_500,
    }


def test_chroma_numeric_filter_is_inclusive(numeric_filter: IntFilter):
    adapter_cls = _adapter_class(
        "chromadb",
        "vectordb_bench.backend.clients.chroma.chroma",
        "ChromaClient",
    )

    assert FilterOp.NumGE in adapter_cls.supported_filter_types
    adapter = adapter_cls.__new__(adapter_cls)
    adapter.prepare_filter(numeric_filter)
    assert adapter._where_filter == {"index": {"$gte": 49_500}}


def test_elasticsearch_numeric_filter_is_inclusive(numeric_filter: IntFilter):
    adapter_cls = _adapter_class(
        "elasticsearch",
        "vectordb_bench.backend.clients.elastic_cloud.elastic_cloud",
        "ElasticCloud",
    )

    assert FilterOp.NumGE in adapter_cls.supported_filter_types
    adapter = adapter_cls.__new__(adapter_cls)
    adapter.id_col_name = "id"
    adapter.prepare_filter(numeric_filter)
    assert adapter.filter == {"range": {"id": {"gte": 49_500}}}
