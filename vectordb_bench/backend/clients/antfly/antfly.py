import logging
import math
import time
from contextlib import contextmanager
from typing import Any

import httpx

from ..api import DBCaseConfig, MetricType, VectorDB

log = logging.getLogger(__name__)

BATCH_CHUNK_SIZE = 500
TABLE_READY_TIMEOUT = 30
TABLE_READY_POLL_INTERVAL = 2
INDEX_READY_TIMEOUT = 1800
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

        base_url = f"http://{_httpx_host(db_config['host'])}:{db_config['port']}/api/v1"
        self._base_url = base_url
        num_shards = db_config.get("num_shards", 1)

        client = httpx.Client(base_url=base_url, timeout=60)
        try:
            if drop_old:
                r = client.delete(f"/tables/{self.collection_name}")
                log.info(f"Drop table response: {r.status_code}")

            # Two-step table creation to avoid Pebble lock issue:
            # 1. Create table without embeddings index
            r = client.post(f"/tables/{self.collection_name}", json={"num_shards": num_shards})
            log.info(f"Create table response: {r.status_code}")

            # Wait for shard to initialize
            self._wait_for_shard_ready(client)

            # 2. Add field-only embeddings index (pre-computed vectors, no embedder needed)
            index_def = {
                "name": INDEX_NAME,
                "dimension": dim,
                "external": True,
                **self.case_config.index_param(),
            }
            index_error = None
            for index_type in INDEX_TYPES:
                r = client.post(
                    f"/tables/{self.collection_name}/indexes/{INDEX_NAME}",
                    json={"type": index_type, **index_def},
                )
                log.info(f"Add embeddings index response ({index_type}): {r.status_code}")
                if r.is_success:
                    index_error = None
                    break
                index_error = r
            if index_error is not None:
                index_error.raise_for_status()
            self._wait_for_index_ready(client, expected_total=0)
        finally:
            client.close()

    def _wait_for_shard_ready(self, client: httpx.Client):
        """Wait for shard to be initialized and accepting writes."""
        deadline = time.monotonic() + TABLE_READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = client.post(
                    f"/tables/{self.collection_name}/batch",
                    json={"inserts": {"_healthcheck": {"_probe": True}}, "sync_level": "write"},
                )
                if r.status_code < 500:
                    # Delete the probe doc
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
        self.client = httpx.Client(base_url=self._base_url, timeout=120)
        try:
            yield
        finally:
            self.client.close()
            self.client = None

    def need_normalize_cosine(self) -> bool:
        # TaskRunner already gates normalization on the dataset metric being COSINE.
        # Returning True here avoids relying on mutable case_config state to decide
        # whether Antfly should receive unit-normalized vectors.
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

    def ready_to_search(self) -> bool:
        if getattr(self, "client", None) is not None:
            payload = self._get_index_status(self.client)
            return self._index_status_is_ready(payload, payload.get("status") if payload else None)
        with httpx.Client(base_url=self._base_url, timeout=120) as client:
            payload = self._get_index_status(client)
            return self._index_status_is_ready(payload, payload.get("status") if payload else None)

    def optimize(self, data_size: int | None = None):
        if getattr(self, "client", None) is not None:
            self._wait_for_index_ready(self.client, expected_total=data_size)
            return
        with httpx.Client(base_url=self._base_url, timeout=120) as client:
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
                    inserts[key] = {
                        "id": metadata[i],
                        "metadata": metadata[i],
                        # Antfly derives a hashID for precomputed embeddings from the
                        # configured source field. Give it a stable string value instead
                        # of forcing the "missing field" path for every document.
                        SOURCE_FIELD: str(metadata[i]),
                        "_embeddings": {"vec": embedding},
                    }
                payload = {"inserts": inserts, "sync_level": "aknn"}
                r = self.client.post(f"/tables/{self.collection_name}/batch", json=payload)
                if not r.is_success:
                    log.warning(
                        "Antfly aknn batch write failed (%s), falling back to sync_level=write: %s",
                        r.status_code,
                        r.text,
                    )
                    r = self.client.post(
                        f"/tables/{self.collection_name}/batch",
                        json={"inserts": inserts, "sync_level": "write"},
                    )
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
        # Workaround: omit semantic_search and indexes, send embeddings directly.
        # This bypasses generateQueryEmbeddings() and uses the direct vector path.
        if self._uses_cosine_distance():
            query = self._normalize_vector(query)
        body = {
            "embeddings": {"vec": query},
            "limit": k,
            "fields": ["id"],
        }
        r = self.client.post(
            f"/tables/{self.collection_name}/query",
            json=body,
        )
        r.raise_for_status()
        data = r.json()
        # Response format: {"responses": [{"hits": {"hits": [{"_id": "key:42", ...}, ...]}}]}
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
