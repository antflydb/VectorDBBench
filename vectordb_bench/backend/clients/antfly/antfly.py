import base64
import json
import logging
import math
import os
import struct
import time
from contextlib import contextmanager
from typing import Any

import httpx

from ..api import DBCaseConfig, MetricType, VectorDB
from ...filter import Filter, FilterOp
from ...payload import PayloadProfile

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


def _pack_dense_f32(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def _make_client(base_url: str, timeout: float) -> httpx.Client:
    raw_timeout = os.environ.get("ANTFLY_VDBBENCH_HTTP_TIMEOUT")
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            log.warning("Ignoring invalid ANTFLY_VDBBENCH_HTTP_TIMEOUT=%r", raw_timeout)
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        limits=httpx.Limits(max_keepalive_connections=8, max_connections=8),
    )


def _detect_api_root(host: str, port: int) -> str:
    # Prefer the current antfly-zig public API root, falling back to the
    # legacy /api/v1 root served by Go binaries. The content-type check
    # matters: legacy binaries answer unknown /db/v1 paths with the dashboard
    # SPA (200 text/html), not a 404.
    for root in ("/db/v1", "/api/v1"):
        try:
            r = httpx.get(f"http://{host}:{port}{root}/tables", timeout=5)
        except httpx.HTTPError:
            continue
        if r.status_code < 500 and "json" in r.headers.get("content-type", ""):
            return root
    return "/db/v1"


class Antfly(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "vdbbench",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.dim = dim
        self.with_scalar_labels = with_scalar_labels
        self._filter_query: dict[str, Any] | None = None

        # Antfly v0.1 used /api/v1; current antfly-zig serves the public DB API
        # at /db/v1. Auto-detect unless ANTFLY_API_ROOT pins it explicitly.
        api_root = os.environ.get("ANTFLY_API_ROOT", "").rstrip("/")
        if not api_root:
            api_root = _detect_api_root(
                _httpx_host(db_config["host"]), db_config["port"]
            )
            log.info(f"Detected Antfly API root: {api_root}")
        self._legacy_api = api_root == "/api/v1"
        self._metadata_base_url = (
            f"http://{_httpx_host(db_config['host'])}:{db_config['port']}{api_root}"
        )
        self._store_host = _httpx_host(db_config.get("store_host") or db_config["host"])
        self._store_port = db_config.get("store_port")
        self._use_direct_store_search = bool(db_config.get("use_direct_store_search"))
        self._pack_query_vectors = bool(db_config.get("pack_query_vectors"))
        self._write_sync_level = os.environ.get(
            "ANTFLY_VDBBENCH_SYNC_LEVEL",
            "write",
        )
        # TEMPORARY client-side backpressure for async sync levels, until the
        # server applies its own under sustained ingest. With sync_level
        # "write" nothing throttles inserts, and a sustained 1M load outruns
        # dense catch-up + LSM compaction until the write path stalls
        # (observed as insert timeouts around 200-600k docs). Pause between
        # batches whenever the server-reported catch-up backlog exceeds
        # max_lag sequences (~100 docs per sequence), resuming at resume_lag.
        # Remove once the server keeps catch-up healthy on its own.
        self._pace_max_lag = int(os.environ.get("ANTFLY_VDBBENCH_MAX_LAG_SEQ", "200"))
        self._pace_resume_lag = int(os.environ.get("ANTFLY_VDBBENCH_RESUME_LAG_SEQ", "50"))
        self._pace_check_every = int(os.environ.get("ANTFLY_VDBBENCH_PACE_EVERY", "5"))
        self._pace_batches_since_check = 0
        self._direct_shard_id: str | None = None
        self._bench_status_last_log = 0.0
        num_shards = db_config.get("num_shards", 1)

        if self._use_direct_store_search and not self._store_port:
            raise ValueError(
                "Antfly direct store search requires store_port to be configured"
            )

        client = _make_client(self._metadata_base_url, 60)
        try:
            if drop_old:
                r = client.delete(f"/tables/{self.collection_name}")
                log.info(f"Drop table response: {r.status_code}")

            table = self._get_table_status_or_none(client)
            if table is None:
                r = client.post(
                    f"/tables/{self.collection_name}", json={"num_shards": num_shards}
                )
                log.info(f"Create table response: {r.status_code}")
                r.raise_for_status()
                self._wait_for_shard_ready(client)
                # Wait for the write path before touching indexes: legacy (Go)
                # binaries can permanently orphan a shard if an index-add lands
                # while the shard is still initializing.
                self._wait_for_write_ready(client)
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
                # both old binaries (require field) and new source (reject
                # field with external). Order matters: legacy binaries accept
                # the field-less variant at the metadata layer but shard-level
                # registration then fails forever ("field or template must be
                # specified"), so on the legacy API try the field variant first.
                if api_root == "/api/v1":
                    field_variants = ({"field": SOURCE_FIELD}, {})
                else:
                    field_variants = ({}, {"field": SOURCE_FIELD})
                for index_type in INDEX_TYPES:
                    for extra in field_variants:
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
            # Do not wait for an empty external index to finish rebuilding here.
            # antfly-zig keeps an empty external dense index in backfill state
            # until writes arrive, and optimize(data_size=...) performs the
            # real post-load readiness wait below.
            self._refresh_direct_search_routing(client)
        finally:
            client.close()

    def _wait_for_shard_ready(self, client: httpx.Client):
        deadline = time.monotonic() + TABLE_READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                table = self._get_table_status_or_none(client)
                if table is not None:
                    log.info("Shard metadata is ready")
                    return
            except Exception as exc:
                log.debug("Shard readiness probe failed", exc_info=exc)
            time.sleep(TABLE_READY_POLL_INTERVAL)
        log.warning(
            f"Shard readiness timeout after {TABLE_READY_TIMEOUT}s, proceeding anyway"
        )

    def _wait_for_write_ready(self, client: httpx.Client):
        # Table metadata appearing does not mean shards accept writes yet:
        # legacy (Go) binaries return 500 "shard is still initializing" for a
        # few seconds after table creation. Probe with a throwaway document
        # (no embedding, so it never lands in the vector index) until a batch
        # write succeeds. The probe doc is intentionally left in place: deleting
        # it would leave a tombstone, and a table that has ever deleted a doc
        # loses the "all docs visible" fast path — every dense query then
        # materializes the full live-doc set as a positive filter, which costs
        # O(table size) per query (~40ms/query at 1M docs).
        probe_key = "key:__circus_write_probe__"
        deadline = time.monotonic() + TABLE_READY_TIMEOUT
        last_error = None
        while time.monotonic() < deadline:
            try:
                r = client.post(
                    f"/tables/{self.collection_name}/batch",
                    json={"inserts": {probe_key: {"id": -1}}, "sync_level": "write"},
                )
                if r.is_success:
                    log.info("Write path is ready")
                    return
                last_error = f"{r.status_code}: {r.text[:120]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(TABLE_READY_POLL_INTERVAL)
        log.warning(
            f"Write readiness timeout after {TABLE_READY_TIMEOUT}s ({last_error}), proceeding anyway"
        )

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

    def _bench_status_enabled(self) -> bool:
        return os.environ.get("ANTFLY_BENCH_STATUS") == "1"

    def _bench_status_interval(self) -> float:
        raw = os.environ.get("ANTFLY_BENCH_STATUS_INTERVAL", "30")
        try:
            return max(float(raw), 1.0)
        except ValueError:
            return 30.0

    def _maybe_log_bench_status(
        self,
        client: httpx.Client,
        phase: str,
        *,
        force: bool = False,
    ) -> None:
        if not self._bench_status_enabled():
            return
        now = time.monotonic()
        if not force and now - self._bench_status_last_log < self._bench_status_interval():
            return
        self._bench_status_last_log = now

        try:
            table = self._get_table_status_or_none(client)
            index = self._get_index_status(client)
            log.info(
                "antfly_bench_status %s",
                json.dumps(
                    {
                        "phase": phase,
                        "table": self._compact_table_status(table),
                        "index": self._compact_index_status(index),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        except Exception as exc:
            log.warning("Antfly bench status probe failed: %s", exc)

    @staticmethod
    def _compact_table_status(table: dict | None) -> dict | None:
        if table is None:
            return None
        shards = table.get("shards") or {}
        storage = table.get("storage_status") or {}
        return {
            "name": table.get("name"),
            "shard_count": len(shards),
            "empty": storage.get("empty"),
            "lsm": storage.get("lsm"),
        }

    @staticmethod
    def _compact_index_status(index: dict | None) -> dict | None:
        if index is None:
            return None
        status = index.get("status") or {}
        async_indexing = status.get("async_indexing") or {}
        dense_catch_up = async_indexing.get("dense_catch_up") or {}
        hbc_cache = status.get("hbc_cache") or {}
        return {
            "type": (index.get("config") or {}).get("type"),
            "rebuilding": status.get("rebuilding"),
            "total_indexed": status.get("total_indexed"),
            "doc_count": status.get("doc_count"),
            "total_nodes": status.get("total_nodes"),
            "query_visible_doc_count": status.get("query_visible_doc_count"),
            "published_doc_count": status.get("published_doc_count"),
            "backfill_state": status.get("backfill_state"),
            "backfill_progress": status.get("backfill_progress"),
            "catch_up_active": status.get("catch_up_active"),
            "catch_up_phase": status.get("catch_up_phase"),
            "catch_up_applied_sequence": status.get("catch_up_applied_sequence"),
            "catch_up_target_sequence": status.get("catch_up_target_sequence"),
            "dense_publish_pending": status.get("dense_publish_pending"),
            "dense_catch_up": {
                "phase": dense_catch_up.get("phase"),
                "current_sequence": dense_catch_up.get("current_sequence"),
                "current_target_sequence": dense_catch_up.get("current_target_sequence"),
                "finish_calls": dense_catch_up.get("finish_calls"),
                "finish_ns": dense_catch_up.get("finish_ns"),
                "finalize_ns": dense_catch_up.get("finalize_ns"),
                "maintenance_steps": dense_catch_up.get("maintenance_steps"),
                "maintenance_ns": dense_catch_up.get("maintenance_ns"),
                "manifest_writes": dense_catch_up.get("manifest_writes"),
                "write_pressure_compactions": dense_catch_up.get("write_pressure_compactions"),
                "write_pressure_ns": dense_catch_up.get("write_pressure_ns"),
            },
            "hbc_cache_total_bytes": hbc_cache.get("total_bytes"),
        }

    def _refresh_direct_search_routing(self, client: httpx.Client):
        if not self._use_direct_store_search:
            return
        table = self._get_table_status(client)
        shards = table.get("shards") or {}
        if len(shards) != 1:
            msg = f"Antfly direct store search currently requires exactly one shard; found {len(shards)} shards"
            raise ValueError(msg)
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

    def _wait_for_index_ready(
        self, client: httpx.Client, expected_total: int | None = None
    ):
        deadline = time.monotonic() + INDEX_READY_TIMEOUT
        last_status = None

        while time.monotonic() < deadline:
            try:
                payload = self._get_index_status(client)
                status = payload.get("status") if payload else None
                last_status = status
                self._maybe_log_bench_status(client, "optimize_wait")
                if self._index_status_is_ready(payload, status, expected_total):
                    log.info(
                        "Embeddings index is ready: %s",
                        self._compact_index_status(payload),
                    )
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
        self.client = _make_client(self._metadata_base_url, 120)
        self.store_client = None
        try:
            if self._use_direct_store_search:
                self.store_client = _make_client(self._store_base_url, 120)
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
            raise ValueError(
                "Antfly store_base_url requested without store_port configured"
            )
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

    def _serialize_insert_vector(self, vector: list[float]) -> str | list[float]:
        # Legacy 0.1.0 binaries reject the packed base64 format on writes
        # ("embedding ... must be an array (dense) or object (sparse), got
        # string"); plain JSON arrays are accepted by every version.
        if self._legacy_api:
            return vector
        return self._pack_vector(vector)

    def _metadata_query_body(self, query: list[float], k: int) -> dict[str, Any]:
        body = {
            "embeddings": {"vec": self._serialize_query_vector(query)},
            "limit": k,
            "fields": [],
            **self.case_config.search_param(),
        }
        if getattr(self, "_filter_query", None) is not None:
            body["filter_query"] = self._filter_query
        return body

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
            return self._index_status_is_ready(
                payload, payload.get("status") if payload else None
            )
        with _make_client(self._metadata_base_url, 120) as client:
            payload = self._get_index_status(client)
            return self._index_status_is_ready(
                payload, payload.get("status") if payload else None
            )

    def optimize(self, data_size: int | None = None):
        if getattr(self, "client", None) is not None:
            self._maybe_log_bench_status(self.client, "optimize_start", force=True)
            self._wait_for_index_ready(self.client, expected_total=data_size)
            self._maybe_log_bench_status(self.client, "optimize_end", force=True)
            return
        with _make_client(self._metadata_base_url, 120) as client:
            self._maybe_log_bench_status(client, "optimize_start", force=True)
            self._wait_for_index_ready(client, expected_total=data_size)
            self._maybe_log_bench_status(client, "optimize_end", force=True)

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self._filter_query = None
        elif filters.type == FilterOp.NumGE:
            self._filter_query = {
                "numeric_range": {
                    "field": filters.int_field,
                    "min": filters.int_value,
                    "inclusive_min": True,
                }
            }
        elif filters.type == FilterOp.StrEqual:
            self._filter_query = {
                "term": {filters.label_field: filters.label_value}
            }
        else:
            raise ValueError(f"Unsupported Antfly filter: {filters}")

    def _catch_up_lag_sequences(self) -> int | None:
        try:
            payload = self._get_index_status(self.client)
            status = (payload or {}).get("status") or {}
            applied = status.get("catch_up_applied_sequence")
            target = status.get("catch_up_target_sequence")
            if applied is None or target is None:
                return None
            return max(0, int(target) - int(applied))
        except Exception:
            return None

    def _pace_async_indexing(self) -> None:
        if self._pace_max_lag <= 0 or self._write_sync_level == "full_index":
            return
        self._pace_batches_since_check += 1
        if self._pace_batches_since_check < self._pace_check_every:
            return
        self._pace_batches_since_check = 0
        lag = self._catch_up_lag_sequences()
        if lag is None or lag <= self._pace_max_lag:
            return
        log.info("Antfly pacing: catch-up lag %d sequences, waiting", lag)
        while lag is not None and lag > self._pace_resume_lag:
            time.sleep(1)
            lag = self._catch_up_lag_sequences()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
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
                    serialized_embedding = self._serialize_insert_vector(embedding)
                    inserts[key] = {
                        "id": metadata[i],
                        "metadata": metadata[i],
                        SOURCE_FIELD: str(metadata[i]),
                        "_embeddings": {"vec": serialized_embedding},
                    }
                    if self.with_scalar_labels:
                        if labels_data is None:
                            raise ValueError("Antfly label-filter load requires labels_data")
                        inserts[key]["labels"] = labels_data[i]
                payload = {"inserts": inserts, "sync_level": self._write_sync_level}
                r = self.client.post(
                    f"/tables/{self.collection_name}/batch", json=payload
                )
                r.raise_for_status()
                self._maybe_log_bench_status(self.client, "insert")
                self._pace_async_indexing()
        except Exception as e:
            log.warning(f"Antfly insert error: {e}")
            return 0, e
        return total, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        if payload_profile != PayloadProfile.IDS_ONLY:
            raise NotImplementedError(
                f"Antfly VDBBench adapter only supports payload_profile={PayloadProfile.IDS_ONLY.value}"
            )
        if self._uses_cosine_distance():
            query = self._normalize_vector(query)

        if self._use_direct_store_search:
            if self._filter_query is not None:
                raise ValueError("Antfly filtered ANN requires the public metadata query API")
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
