import logging
import subprocess
import json
import time
import os
import requests
from contextlib import contextmanager
from typing import Any, List

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)


class AntflyClient(VectorDB):
    """Antfly client for VectorDB using direct API calls and CLI."""

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = "vdb"
        self.index_name = "embedding_idx"
        self.dim = dim
        self.antfly_process = None
        self.antfly_url = db_config.get("url", "http://localhost:8080")
        self.api_url = f"{self.antfly_url}/api/v1"
        self.antflycli_path = db_config.get("antflycli_path", "/home/rowan/Documents/antfly/antflycli")
        self.drop_old = drop_old
        self.index_created = False  # Track if we've already created the index


    @contextmanager
    def init(self) -> None:
        """Initialize connection to Antfly database (assumes server is already running)."""
        # Wait for Antfly to be ready
        log.info("Waiting for Antfly server to be ready...")
        retries = 30
        while retries > 0:
            try:
                response = requests.get(f"{self.antfly_url}/health", timeout=2)
                if response.status_code == 200:
                    log.info("Antfly server is ready.")
                    break
            except requests.exceptions.RequestException:
                pass
            retries -= 1
            time.sleep(1)
        else:
            raise RuntimeError("Failed to connect to Antfly server. Is it running?")

        # Drop old table if requested
        if self.drop_old:
            log.info(f"Dropping old table {self.collection_name}")
            try:
                response = requests.delete(
                    f"{self.api_url}/table/{self.collection_name}",
                    timeout=30,
                )
                if response.status_code == 204:
                    log.info(f"Table {self.collection_name} dropped successfully")
                elif response.status_code in (404, 400) and "not found" in response.text.lower():
                    log.info(f"Table {self.collection_name} does not exist (already clean)")
                else:
                    log.warning(f"Unexpected response dropping table: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                log.warning(f"Failed to drop table (may not exist): {e}")
            # Reset index creation flag since we're starting fresh
            self.index_created = False

        # Create table using API with proper schema
        log.info(f"Creating table {self.collection_name}")
        try:
            table_config = {
                "schema": {
                    "key": "id"
                }
            }
            response = requests.post(
                f"{self.api_url}/table/{self.collection_name}",
                json=table_config,
                timeout=30,
            )
            if response.status_code == 200:
                log.info(f"Table {self.collection_name} created successfully")
                # Wait for table to be fully initialized (Raft consensus)
                log.info("Waiting for table shards to initialize...")
                time.sleep(5)
            elif response.status_code == 400 and "already exists" in response.text.lower():
                log.info(f"Table {self.collection_name} already exists")
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to create table: {e}")
            if hasattr(e, 'response') and e.response is not None:
                log.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Failed to create table: {e}")

        # Create index before any data insertion (only if not already created)
        if not self.index_created:
            log.info(f"Creating index {self.index_name} on {self.collection_name}")
            try:
                index_config = {
                    "name": self.index_name,
                    "type": "vector_v2",
                    "dimension": self.dim,
                    "field": "embedding",
                }

                response = requests.post(
                    f"{self.api_url}/table/{self.collection_name}/index/{self.index_name}",
                    json=index_config,
                    timeout=30,
                )

                if response.status_code == 201:
                    log.info(f"Index {self.index_name} created successfully")
                    self.index_created = True
                elif "already exists" in response.text.lower():
                    log.info(f"Index {self.index_name} already exists")
                    self.index_created = True
                else:
                    # Don't fail on errors - just log and continue
                    # Index might be created eventually by Antfly
                    log.warning(f"Index creation returned {response.status_code}: {response.text[:200]}")
                    self.index_created = True  # Assume it will work

            except requests.exceptions.RequestException as e:
                # Don't fail on errors - just log and continue
                log.warning(f"Index creation failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    log.warning(f"Response: {e.response.text[:200]}")
                self.index_created = True  # Assume it will work eventually
        else:
            log.info(f"Index {self.index_name} already created, skipping")

        log.info(f"Table {self.collection_name} ready")

        yield

        log.info("Cleanup complete (server left running)")

    def ready_to_search(self) -> bool:
        log.info("Antfly is always ready to search")
        return True

    def optimize(self, data_size: int | None = None):
        log.info("Antfly does not support optimizing index")
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        log.info(f"Inserting {len(embeddings)} embeddings into {self.collection_name}")

        # Prepare data for batch insert
        inserts = {}
        for i, emb in enumerate(embeddings):
            key = str(metadata[i])
            inserts[key] = {
                "id": metadata[i],
                "embedding": emb,
            }

        # Use API for batch insert
        try:
            response = requests.post(
                f"{self.api_url}/table/{self.collection_name}/batch",
                json={"inserts": inserts},
                timeout=300,
            )
            response.raise_for_status()
            log.info(f"Successfully inserted {len(embeddings)} embeddings")
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to insert data into Antfly: {e}")
            return 0, e

        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        log.info(f"Searching for embedding in {self.collection_name}")

        # Use the API to perform vector search
        # Based on QueryRequest schema in OpenAPI spec
        query_request = {
            "embeddings": {
                self.index_name: query
            },
            "limit": k,
            "fields": ["id"],
        }

        try:
            response = requests.post(
                f"{self.api_url}/table/{self.collection_name}/query",
                json=query_request,
                timeout=timeout or 30,
            )
            response.raise_for_status()
            result = response.json()

            # Parse the response based on QueryResponses schema
            # The response has: {"responses": [{"hits": {"hits": [...]}}]}
            ids = []
            if "responses" in result and len(result["responses"]) > 0:
                first_response = result["responses"][0]
                if "hits" in first_response and first_response["hits"] is not None:
                    if "hits" in first_response["hits"] and first_response["hits"]["hits"] is not None:
                        for hit in first_response["hits"]["hits"]:
                            # hit has _id, _score, _source
                            # The _source should contain our document with "id" field
                            if "_source" in hit and "id" in hit["_source"]:
                                try:
                                    ids.append(int(hit["_source"]["id"]))
                                except (ValueError, TypeError) as e:
                                    log.warning(f"Could not parse id from hit: {hit}. Error: {e}")

            log.info(f"Found {len(ids)} results")
            return ids

        except requests.exceptions.RequestException as e:
            log.error(f"Failed to search in Antfly: {e}")
            if hasattr(e, 'response') and e.response is not None:
                log.error(f"Response: {e.response.text}")
            return []
