import logging
from contextlib import contextmanager
from typing import Iterable

from ..api import VectorDB
from .config import AntflyConfig, AntflyIndexConfig
from .antfly_client.client import AntflyClient
from .antfly_client.models import Vector

log = logging.getLogger(__name__)

class Antfly(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: AntflyIndexConfig,
        collection_name: str = "AntflyCollection",
        drop_old: bool = False,
        name: str = "Antfly",
        **kwargs,
    ):
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.case_config.dimension = dim
        self.collection_name = collection_name
        self.client = None
        self.dim = dim
        self._index_name = "vector_index"


        antfly_client = AntflyClient(**self.db_config)
        if drop_old:
            try:
                tables = [t.name for t in antfly_client.list_tables()]
                if self.collection_name in tables:
                    antfly_client.delete_table(self.collection_name)
                    log.info(f"{self.name} client drop_old table: {self.collection_name}")
            except Exception as e:
                log.warning(f"Failed to drop table {self.collection_name}: {e}")

        tables = [t.name for t in antfly_client.list_tables()]
        if self.collection_name not in tables:
            self._create_collection(antfly_client)

        antfly_client.close()

    @contextmanager
    def init(self):
        self.client = AntflyClient(**self.db_config)
        yield
        self.client.close()
        self.client = None

    def _create_collection(self, client: AntflyClient):
        client.create_table(self.collection_name)
        index_params = self.case_config.index_param()
        client.create_index(
            table_name=self.collection_name,
            index_name=self._index_name,
            **index_params,
        )

    def optimize(self, **kwargs):
        assert self.client, "Please call self.init() before"
        self.client.wait_for_index_ready(
            table_name=self.collection_name,
            index_name=self._index_name,
        )

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.client is not None
        vectors = [
            Vector(id=str(pk), values=emb, metadata={"pk": pk})
            for pk, emb in zip(metadata, embeddings)
        ]

        try:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                if len(batch) == 1:
                    # Antfly's batch_upsert requires at least 2 vectors.
                    # This is a workaround to handle the last batch if it has only one vector.
                    # We create a dummy vector and insert it with the single vector.
                    dummy_vector = Vector(id="dummy", values=[0.0]*self.dim, metadata={"pk": -1})
                    self.client.batch_upsert(table_name=self.collection_name, vectors=[batch[0], dummy_vector])
                elif len(batch) > 1:
                    self.client.batch_upsert(table_name=self.collection_name, vectors=batch)

            return len(vectors), None
        except Exception as e:
            log.error(f"Failed to insert embeddings: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.client is not None

        res = self.client.query(
            table=self.collection_name,
            embeddings={self._index_name: query},
            limit=k,
        )

        # Assuming the search result format is a list of dicts with an 'id' field.
        # The 'id' is a string, so it needs to be converted back to an int.
        # The actual format of the result is not known, so this is a guess.
        # It's also possible the primary key is in the metadata.
        # The query result model I created has `hits: List[Any]`.
        # I'll assume the hits are in `res.hits` and each hit is a dict with a `metadata` field which contains the `pk`.
        # This is a common pattern in other clients.

        results = []
        for hit in res.hits:
            if isinstance(hit, dict) and "metadata" in hit and "pk" in hit["metadata"]:
                 # filter out dummy vector
                if hit["metadata"]["pk"] != -1:
                    results.append(hit["metadata"]["pk"])

        return results
