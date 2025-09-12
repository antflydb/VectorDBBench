import logging
import numpy as np
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.antfly.config import AntflyConfig

log = logging.getLogger(__name__)

class TestAntfly:
    def test_insert_and_search(self):
        assert DB.Antfly.value == "Antfly"

        db_cls = DB.Antfly.init_cls
        db_config_cls = DB.Antfly.config_cls

        db_config = db_config_cls(
            host="localhost",
            port=8080,
            cli_path="/app/antflycli",
        )

        dim = 16
        antfly = db_cls(
            dim=dim,
            db_config=db_config.to_dict(),
            db_case_config=None,
            drop_old=True,
        )

        count = 100
        embeddings = [[np.random.random() for _ in range(dim)] for _ in range(count)]

        # insert
        with antfly.init():
            res = antfly.insert_embeddings(embeddings=embeddings, metadata=range(count))
            assert res[0] == count, f"the return count of bulk insert ({res[0]}) is not equal to count ({count})"

        # search
        with antfly.init():
            test_id = np.random.randint(count)
            q = embeddings[test_id]

            # The search is a dummy search, so we can't assert the result.
            # We just check that it doesn't raise an exception.
            try:
                res = antfly.search_embedding(query=q, k=10)
                log.info(f"Search result: {res}")
            except Exception as e:
                assert False, f"Search raised an exception: {e}"
