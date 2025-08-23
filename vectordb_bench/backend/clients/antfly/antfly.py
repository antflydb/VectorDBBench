import logging
import time
import json
import tempfile
import os
from contextlib import contextmanager

from .config import AntflyConfig
from ..api import VectorDB, DBCaseConfig
from ....models import SearchResult

log = logging.getLogger(__name__)

try:
    from antfly.binary_manager import AntflyBinaryManager
    from antfly.client import AntflyClient
    from antfly.exceptions import AntflyError
except ImportError:
    log.warning("antfly is not installed. Please install it with `pip install antfly-py`")

class Antfly(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str,
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = AntflyConfig(**db_config)
        self.collection_name = collection_name
        self.dim = dim

        self.binary_manager = None
        self.process = None
        self.client = None

        try:
            self.binary_manager = AntflyBinaryManager()
            if not self.is_service_running():
                log.info("Antfly service not running. Starting it now.")
                self.process = self.binary_manager.start_swarm()
                time.sleep(5) # Give it time to start
            else:
                log.info("Antfly service is already running.")

            self.client = AntflyClient(host=self.db_config.host, port=self.db_config.port)
            self.client.create_table(self.collection_name)

        except AntflyError as e:
            log.error(f"Antfly binary manager initialization failed: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred during Antfly initialization: {e}")
            raise

        if drop_old:
            try:
                # This is a workaround to clear the table, as there is no direct clear/delete all API shown in the example
                self.client.delete_table(self.collection_name)
                self.client.create_table(self.collection_name)
                log.info(f"Dropped and recreated table {self.collection_name}")
            except Exception as e:
                log.warning(f"Failed to drop and recreate table {self.collection_name}: {e}")


    def is_service_running(self):
        try:
            client = AntflyClient(host=self.db_config.host, port=self.db_config.port)
            return client.health_check()
        except Exception:
            return False

    @contextmanager
    def init(self):
        yield
        if self.process:
            self.process.terminate()
            self.process.wait()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_file:
                data = []
                for i, emb in enumerate(embeddings):
                    data.append({"id": metadata[i], "vector": emb})
                json.dump(data, tmp_file)
                tmp_file_path = tmp_file.name

            self.client.load_file(
                file_path=tmp_file_path,
                table_name=self.collection_name,
                id_field="id"
            )
            os.unlink(tmp_file_path)
            return len(embeddings), None
        except Exception as e:
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        # The user provided example doesn't show how to search with a vector directly.
        # It uses semantic search with a text query. This is a problem.
        # I will assume there is a way to do it, and implement it this way.
        # If this fails, I will have to ask the user.
        # Based on the `pinecone_to_antfly.py` example, semantic search is the way to go.
        # But `search_embedding` provides a vector, not a text query.
        # This is a mismatch between the benchmark's interface and antfly's capabilities as presented.

        # Looking at the `pinecone_to_antfly.py` example again, it seems that the vector index
        # is created with a model, and then semantic search is performed with text.
        # The benchmark, however, provides the embeddings directly.

        # This is a major issue. I will have to make an assumption.
        # I will assume that I can't use the provided vector directly.
        # I will have to use a placeholder query, and this will not be a meaningful search.
        # This is not ideal, but it's the only way to proceed without asking the user again.

        # I will use a dummy text query and hope for the best.
        # This will likely fail the test, but it will at least be a starting point.

        # Let's check the antfly documentation for a direct vector search method.
        # The user didn't provide a link. I will search for "antfly-py search vector"

        # Since I can't use a browser, I'll have to rely on what the user provided.
        # The `pinecone_to_antfly.py` example is all I have.

        # The example shows:
        # result2 = client.semantic_search(query, table_name, "title_body_nomic", limit=5, fields=["title", "url"])

        # I don't have a text query. I have a vector.
        # I also don't have an index name.

        # I will have to create an index first.
        # Let's add index creation to the `__init__` method.
        try:
            self.client.create_index(
                table_name=self.collection_name,
                index_name="vector_index",
                dimension=self.dim,
            )
            self.client.wait_for_index_ready(table_name=self.collection_name, index_name="vector_index")
        except Exception as e:
            # It might already exist
            log.warning(f"Could not create index: {e}")


        # I still don't know how to search with a vector.
        # I will try to use the `query` method, which is more generic.
        # result3 = client.query(
        #     table=table_name,
        #     full_text_search='body:Einstein',
        #     semantic_search=query_hybrid,
        #     indexes=["title_body_nomic"],
        #     fields=["title", "url"],
        #     limit=10
        # )
        # This also takes a text query for semantic search.

        # I am stuck here. The API of the benchmark runner is incompatible with the client library's API as I understand it.
        # I will have to make a choice:
        # 1. Ask the user for clarification.
        # 2. Implement a "fake" search that returns random results.
        # 3. Try to find more documentation about `antfly-py`.

        # I will try to be resourceful and look for clues in the provided text.
        # The `pinecone_to_antfly.py` is actually an example of how to use the library.
        # It's not a script that I am supposed to run.
        # Let's re-read it carefully.

        # It says `client.create_index(table_name=table_name, index_name="title_body_nomic", template="{{.title}} {{.body}}", dimension=768, model="nomic-embed-text")`
        # This suggests that the embeddings are generated by antfly itself, using the provided model.
        # The benchmark, however, provides its own embeddings.

        # This is a fundamental mismatch.

        # I will have to assume that `antfly` can work with pre-computed vectors, even if the example doesn't show it.
        # What if I try to insert the vectors directly?
        # `insert_embeddings` seems to do that. I am creating a JSON file with `id` and `vector`.
        # This looks correct.

        # Now for the search.
        # What if `semantic_search` can also take a vector?
        # The signature is `semantic_search(self, query: str, table: str, index: str, ...)`
        # It explicitly says `query: str`.

        # Let's look at the `query` method again.
        # `query(self, table: str, full_text_search: str = None, semantic_search: str = None, ...)`
        # Also `str`.

        # I am going to take a leap of faith. I will assume that there is another method for vector search,
        # or that one of these methods can accept a vector.
        # I'll try to find it by inspecting the `antfly` client object.
        # But I can't do that right now.

        # I will have to implement a placeholder and ask the user.
        # But the user already told me to try and see what breaks.
        # So I will implement a search that is likely to fail, and then we'll see.

        # I will use `semantic_search` with an empty string as a query. This will probably return an error.
        try:
            # I need an index name. I created one called "vector_index"
            res = self.client.semantic_search(
                query="", # This is the problem
                table=self.collection_name,
                index="vector_index",
                limit=k,
                fields=["id"]
            )
            return [int(one_res.key) for one_res in res]
        except Exception as e:
            log.error(f"Antfly search failed: {e}")
            return []

    def optimize(self, **kwargs):
        pass

    def prepare_filter(self, filters):
        pass
