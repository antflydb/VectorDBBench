import base64
import struct
from typing import Any

import pytest

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.antfly.antfly import Antfly
from vectordb_bench.backend.clients.antfly.config import AntflyConfig, AntflyIndexConfig
from vectordb_bench.backend.filter import IntFilter, LabelFilter, non_filter
from vectordb_bench.backend.payload import PayloadProfile


class _CaseConfig:
    def index_param(self):
        return {"distance_metric": "cosine"}

    def search_param(self):
        return {"search_effort": 0.6}


class _Response:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self.text = ""

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self):
        self.posts = []

    def post(self, path: str, *, json: dict[str, Any]):
        self.posts.append((path, json))
        return _Response()


def _adapter(*, with_scalar_labels: bool = False):
    adapter = Antfly.__new__(Antfly)
    adapter.case_config = _CaseConfig()
    adapter.collection_name = "vdbbench"
    adapter.with_scalar_labels = with_scalar_labels
    adapter._filter_query = None
    adapter._pack_query_vectors = True
    adapter._legacy_api = False
    adapter._write_sync_level = "write"
    adapter._bench_status_last_log = 0.0
    adapter.client = _Client()
    return adapter


def test_antfly_is_registered():
    assert DB.Antfly.config_cls is AntflyConfig
    assert DB.Antfly.case_config_cls() is AntflyIndexConfig
    assert DB.Antfly.init_cls is Antfly


def test_antfly_uses_native_cosine_and_ids_only_queries():
    adapter = _adapter()

    assert adapter.need_normalize_cosine() is False
    body = adapter._metadata_query_body([1.0, 2.0], 10)
    assert body["fields"] == []
    assert body["search_effort"] == 0.6

    adapter.prepare_filter(IntFilter(filter_rate=0.99, int_field="id", int_value=49_500))
    assert adapter._metadata_query_body([1.0, 2.0], 10)["filter_query"] == {
        "numeric_range": {"field": "id", "min": 49_500, "inclusive_min": True}
    }

    adapter.prepare_filter(LabelFilter(label_percentage=0.01))
    assert adapter._filter_query == {"term": {"labels": "label_1p"}}
    adapter.prepare_filter(non_filter)
    assert "filter_query" not in adapter._metadata_query_body([1.0, 2.0], 10)

    with pytest.raises(NotImplementedError):
        adapter.search_embedding([1.0, 2.0], payload_profile=PayloadProfile.VECTOR)


def test_antfly_insert_preserves_vectors_and_labels():
    adapter = _adapter(with_scalar_labels=True)

    inserted, error = adapter.insert_embeddings([[3.0, 4.0]], [7], labels_data=["bucket_007"])

    assert error is None
    assert inserted == 1
    _, payload = adapter.client.posts[0]
    row = payload["inserts"]["key:7"]
    vector = base64.b64decode(row["_embeddings"]["vec"])
    assert struct.unpack("<2f", vector) == pytest.approx((3.0, 4.0))
    assert row["labels"] == "bucket_007"
    assert payload["sync_level"] == "write"


def test_write_readiness_probe_does_not_create_a_tombstone():
    adapter = _adapter()
    client = _Client()

    adapter._wait_for_write_ready(client)

    assert len(client.posts) == 1
    assert client.posts[0][1] == {
        "inserts": {"key:__circus_write_probe__": {"id": -1}},
        "sync_level": "write",
    }


def test_vector_benchmark_removes_default_full_text_index(monkeypatch):
    adapter = _adapter()

    class Client:
        def __init__(self):
            self.present = True
            self.deleted = []

        def get(self, path: str):
            return _Response(200 if self.present else 404)

        def delete(self, path: str):
            self.deleted.append(path)
            self.present = False
            return _Response(204)

    client = Client()
    monkeypatch.setattr("time.sleep", lambda _: None)

    adapter._remove_default_full_text_index(client)

    assert client.deleted == ["/tables/vdbbench/indexes/full_text_index_v0"]


def test_vector_benchmark_can_keep_default_full_text_index(monkeypatch):
    adapter = _adapter()

    monkeypatch.delenv("ANTFLY_VDBBENCH_KEEP_DEFAULT_FULL_TEXT", raising=False)
    assert adapter._keep_default_full_text_index() is False
    monkeypatch.setenv("ANTFLY_VDBBENCH_KEEP_DEFAULT_FULL_TEXT", "yes")
    assert adapter._keep_default_full_text_index() is True
