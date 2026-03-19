import logging
import time
from contextlib import contextmanager
from typing import Any

import httpx

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)

BATCH_CHUNK_SIZE = 500
TABLE_READY_TIMEOUT = 30
TABLE_READY_POLL_INTERVAL = 2


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

        base_url = f"http://{db_config['host']}:{db_config['port']}/api/v1"
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
            index_def = {"type": "embeddings", "dimension": dim, "field": "vec_data"}
            r = client.post(f"/tables/{self.collection_name}/indexes/vec", json=index_def)
            log.info(f"Add embeddings index response: {r.status_code}")
        finally:
            client.close()

    def _wait_for_shard_ready(self, client: httpx.Client):
        """Wait for shard to be initialized and accepting writes."""
        deadline = time.monotonic() + TABLE_READY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = client.post(
                    f"/tables/{self.collection_name}/batch",
                    json={"inserts": {"_healthcheck": {"_probe": True}}},
                )
                if r.status_code < 500:
                    # Delete the probe doc
                    client.post(
                        f"/tables/{self.collection_name}/batch",
                        json={"deletes": ["_healthcheck"]},
                    )
                    log.info("Shard is ready (accepts writes)")
                    return
            except Exception:
                pass
            time.sleep(TABLE_READY_POLL_INTERVAL)
        log.warning(f"Shard readiness timeout after {TABLE_READY_TIMEOUT}s, proceeding anyway")

    @contextmanager
    def init(self):
        self.client = httpx.Client(base_url=self._base_url, timeout=120)
        try:
            yield
        finally:
            self.client.close()
            self.client = None

    def ready_to_search(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        try:
            total = len(embeddings)
            for start in range(0, total, BATCH_CHUNK_SIZE):
                end = min(start + BATCH_CHUNK_SIZE, total)
                inserts = {}
                for i in range(start, end):
                    key = f"key:{metadata[i]}"
                    inserts[key] = {
                        "id": metadata[i],
                        "metadata": metadata[i],
                        "_embeddings": {"vec": embeddings[i]},
                    }
                r = self.client.post(
                    f"/tables/{self.collection_name}/batch",
                    json={"inserts": inserts},
                )
                r.raise_for_status()
            return total, None
        except Exception as e:
            log.warning(f"Antfly insert error: {e}")
            return 0, e

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
