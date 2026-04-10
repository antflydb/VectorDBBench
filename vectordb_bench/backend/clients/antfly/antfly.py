import base64
import logging
import math
import os
import struct
import time
from contextlib import contextmanager
from typing import Any

import httpx

from ..api import DBCaseConfig, MetricType, VectorDB

log = logging.getLogger(__name__)

BATCH_CHUNK_SIZE = 500
TABLE_READY_TIMEOUT = 30
TABLE_READY_POLL_INTERVAL = 2
INDEX_READY_TIMEOUT = 7200
INDEX_READY_POLL_INTERVAL = 2
INDEX_NAME = "vec"
INDEX_TYPES = ("embeddings", "aknn_v0")
SOURCE_FIELD = "vec_data"


def _httpx_host(host: str) -> str:
    # macOS resolves localhost to ::1 first. The current antfly-zig listener is
    # IPv4-only, so keep the user-facing flag but route httpx to IPv4 loopback.
    return "127.0.0.1" if host == "localhost" else host


class Antfly(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "vdbbench",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.dim = dim

        self._metadata_base_url = f"http://{_httpx_host(db_config['host'])}:{db_config['port']}/api/v1"
        self._store_host = _httpx_host(db_config.get("store_host") or db_config["host"])
        self._store_port = db_config.get("store_port")
        self._use_direct_store_search = bool(db_config.get("use_direct_store_search"))
        self._pack_query_vectors = bool(db_config.get("pack_query_vectors"))
        self._direct_shard_id: str | None = None
        num_shards = db_config.get("num_shards", 1)

        if self._use_direct_store_search and not self._store_port:
            raise ValueError("Antfly direct store search requires store_port to be configured")

        client = httpx.Client(base_url=self._metadata_base_url, timeout=60)
        try:
            if drop_old:
                r = client.delete(f"/tables/{self.collection_name}")
                log.info(f"Drop table response: {r.status_code}")

            table = self._get_table_status_or_none(client)
            if table is None:
                r = client.post(f"/tables/{self.collection_name}", json={"num_shards": num_shards})
                log.info(f"Create table response: {r.status_code}")
                r.raise_for_status()
            else:
                log.info("Reusing existing table: %s", self.collection_name)

            self._wait_for_shard_ready(client)

            if self._get_index_status(client) is None:
                index_def = {
                    "name": INDEX_NAME,
                    "dimension": dim,
                    "external": True,
                    **self.case_config.index_param(),
                }
                index_error = None
                # Try each index type, with and without field, to handle
                # both old binaries (require field) and new source (reject field with external).
                for index_type in INDEX_TYPES:
                    for extra in ({}, {"field": SOURCE_FIELD}):
                        r = client.post(
                            f"/tables/{self.collection_name}/indexes/{INDEX_NAME}",
                            json={"type": index_type, **index_def, **extra},
                        )
                        log.info(
                            f"Add embeddings index response ({index_type}, field={'field' in extra}): {r.status_code}"
                        )
                        if r.is_success:
                            index_error = None
                            break
                        index_error = r
                    if index_error is None:
                        break
                if index_error is not None:
                    index_error.raise_for_status()
            else:
                log.info("Reusing existing embeddings index: %s", INDEX_NAME)
            self._wait_for_index_ready(client, expected_total=0)
            self._refresh_direct_search_routing(client)
        finally:
            client.close()

    def _wait_for_shard_ready(self, client: httpx.Client):
        deadline = time.monotonic() + TABLE_READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = client.post(
                    f"/tables/{self.collection_name}/batch",
                    json={"inserts": {"_healthcheck": {"_probe": True}}, "sync_level": "write"},
                )
                if r.status_code < 500:
                    client.post(
                        f"/tables/{self.collection_name}/batch",
                        json={"deletes": ["_healthcheck"], "sync_level": "write"},
                    )
                    log.info("Shard is ready (accepts writes)")
                    return
            except Exception as exc:
                log.debug("Shard readiness probe failed", exc_info=exc)
            time.sleep(TABLE_READY_POLL_INTERVAL)
        log.warning(f"Shard readiness timeout after {TABLE_READY_TIMEOUT}s, proceeding anyway")

    def _get_index_status(self, client: httpx.Client) -> dict | None:
        r = client.get(f"/tables/{self.collection_name}/indexes/{INDEX_NAME}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def _get_table_status(self, client: httpx.Client) -> dict:
        r = client.get(f"/tables/{self.collection_name}")
        r.raise_for_status()
        return r.json()

    def _get_table_status_or_none(self, client: httpx.Client) -> dict | None:
        r = client.get(f"/tables/{self.collection_name}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def _refresh_direct_search_routing(self, client: httpx.Client):
        if not self._use_direct_store_search:
            return
        table = self._get_table_status(client)
        shards = table.get("shards") or {}
        if len(shards) != 1:
            raise ValueError(
                f"Antfly direct store search currently requires exactly one shard; found {len(shards)} shards"
            )
        self._direct_shard_id = next(iter(shards))

    def _index_status_is_ready(
        self,
        payload: dict | None,
        status: dict | None,
        expected_total: int | None = None,
    ) -> bool:
        if payload is None:
            return False
        if status is None:
            return expected_total == 0

        rebuilding = bool(status.get("rebuilding"))
        wal_backlog = int(status.get("wal_backlog", 0) or 0)
        total_indexed = int(status.get("total_indexed", 0) or 0)
        has_error = bool(status.get("error"))

        if has_error or rebuilding or wal_backlog > 0:
            return False
        return expected_total is None or total_indexed >= expected_total

    def _wait_for_index_ready(self, client: httpx.Client, expected_total: int | None = None):
        deadline = time.monotonic() + INDEX_READY_TIMEOUT
        last_status = None

        while time.monotonic() < deadline:
            try:
                payload = self._get_index_status(client)
                status = payload.get("status") if payload else None
                last_status = status
                if self._index_status_is_ready(payload, status, expected_total):
                    log.info(f"Embeddings index is ready: {status}")
                    return
            except Exception as e:
                last_status = {"error": str(e)}
            time.sleep(INDEX_READY_POLL_INTERVAL)

        log.warning(
            "Embeddings index readiness timeout after %ss, expected_total=%s, last_status=%s",
            INDEX_READY_TIMEOUT,
            expected_total,
            last_status,
        )

    @contextmanager
    def init(self):
        self.client = httpx.Client(base_url=self._metadata_base_url, timeout=120)
        self.store_client = None
        try:
            if self._use_direct_store_search:
                self.store_client = httpx.Client(base_url=self._store_base_url, timeout=120)
            yield
        finally:
            self.client.close()
            self.client = None
            if self.store_client is not None:
                self.store_client.close()
                self.store_client = None

    @property
    def _store_base_url(self) -> str:
        if self._store_port is None:
            raise ValueError("Antfly store_base_url requested without store_port configured")
        return f"http://{self._store_host}:{self._store_port}"

    def need_normalize_cosine(self) -> bool:
        return True

    def _uses_cosine_distance(self) -> bool:
        try:
            return self.case_config.index_param().get("distance_metric") == "cosine"
        except Exception:
            return getattr(self.case_config, "metric_type", None) == MetricType.COSINE

    @staticmethod
    def _normalize_vector(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _pack_vector(vector: list[float]) -> str:
        raw = struct.pack(f"<{len(vector)}f", *vector)
        return base64.b64encode(raw).decode("ascii")

    def _serialize_query_vector(self, vector: list[float]) -> list[float] | str:
        if self._pack_query_vectors or os.environ.get("ANTFLY_PACK_VECTORS") == "1":
            return self._pack_vector(vector)
        return vector

    def _metadata_query_body(self, query: list[float], k: int) -> dict[str, Any]:
        return {
            "embeddings": {"vec": self._serialize_query_vector(query)},
            "limit": k,
            "fields": ["id"],
            **self.case_config.search_param(),
        }

    def _store_query_body(self, query: list[float], k: int) -> dict[str, Any]:
        search_params = self.case_config.search_param()
        vector_paging_options: dict[str, Any] = {"limit": k}
        if "search_effort" in search_params:
            vector_paging_options["search_effort"] = search_params["search_effort"]
        return {
            "star": True,
            "limit": k,
            "vector_searches": {INDEX_NAME: self._serialize_query_vector(query)},
            "vector_paging_options": vector_paging_options,
        }

    def _parse_metadata_hits(self, data: dict) -> list[int]:
        resp = data.get("responses", [{}])[0]
        hits_obj = resp.get("hits") or {}
        hits = hits_obj.get("hits") or []
        results = []
        for hit in hits:
            if "id" in hit:
                results.append(int(hit["id"]))
            else:
                doc_key = hit.get("_id", "")
                try:
                    results.append(int(doc_key.split(":", 1)[1]))
                except (IndexError, ValueError):
                    log.warning(f"Could not parse id from _id: {doc_key}")
        return results

    def _parse_store_hits(self, data: dict) -> list[int]:
        vec_result = (data.get("search_result") or {}).get(INDEX_NAME) or {}
        hits = vec_result.get("hits") or []
        results = []
        for hit in hits:
            fields = hit.get("fields") or {}
            if "id" in fields:
                results.append(int(fields["id"]))
                continue
            doc_key = hit.get("id", "")
            try:
                results.append(int(doc_key.split(":", 1)[1]))
            except (IndexError, ValueError):
                log.warning(f"Could not parse id from direct-store hit id: {doc_key}")
        return results

    def ready_to_search(self) -> bool:
        if getattr(self, "client", None) is not None:
            payload = self._get_index_status(self.client)
            return self._index_status_is_ready(payload, payload.get("status") if payload else None)
        with httpx.Client(base_url=self._metadata_base_url, timeout=120) as client:
            payload = self._get_index_status(client)
            return self._index_status_is_ready(payload, payload.get("status") if payload else None)

    def optimize(self, data_size: int | None = None):
        if getattr(self, "client", None) is not None:
            self._wait_for_index_ready(self.client, expected_total=data_size)
            return
        with httpx.Client(base_url=self._metadata_base_url, timeout=120) as client:
            self._wait_for_index_ready(client, expected_total=data_size)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        total = len(embeddings)
        try:
            use_cosine = self._uses_cosine_distance()
            for start in range(0, total, BATCH_CHUNK_SIZE):
                end = min(start + BATCH_CHUNK_SIZE, total)
                inserts = {}
                for i in range(start, end):
                    key = f"key:{metadata[i]}"
                    embedding = embeddings[i]
                    if use_cosine:
                        embedding = self._normalize_vector(embedding)
                    serialized_embedding = self._serialize_query_vector(embedding)
                    inserts[key] = {
                        "id": metadata[i],
                        "metadata": metadata[i],
                        SOURCE_FIELD: str(metadata[i]),
                        "_embeddings": {"vec": serialized_embedding},
                    }
                payload = {"inserts": inserts, "sync_level": "write"}
                r = self.client.post(f"/tables/{self.collection_name}/batch", json=payload)
                r.raise_for_status()
        except Exception as e:
            log.warning(f"Antfly insert error: {e}")
            return 0, e
        return total, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        if self._uses_cosine_distance():
            query = self._normalize_vector(query)

        if self._use_direct_store_search:
            if self._direct_shard_id is None:
                self._refresh_direct_search_routing(self.client)
            r = self.store_client.post(
                "/search",
                headers={"X-Raft-Shard-Id": self._direct_shard_id},
                json=self._store_query_body(query, k),
            )
            r.raise_for_status()
            return self._parse_store_hits(r.json())

        r = self.client.post(
            f"/tables/{self.collection_name}/query",
            json=self._metadata_query_body(query, k),
        )
        r.raise_for_status()
        return self._parse_metadata_hits(r.json())
