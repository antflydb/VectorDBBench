import pytest
from vectordb_bench.backend.clients.antfly.config import AntflyConfig, AntflyIndexConfig
from vectordb_bench.backend.clients.antfly.antfly import Antfly
from vectordb_bench.backend.clients.api import VectorDB

# Note: This test requires an Antfly server running on localhost:8080.
# You can start one by running the command in `vectordb_bench/backend/clients/antfly/cli.py`

DIM = 128
COUNT = 100

db_config = AntflyConfig(
    url="http://localhost:8080",
)

case_config = AntflyIndexConfig(
    dimension=DIM,
)

db = Antfly(
    dim=DIM,
    db_config=db_config.to_dict(),
    db_case_config=case_config,
    collection_name="antfly_test_collection",
    drop_old=True,
)

assert isinstance(db, VectorDB)

@pytest.mark.parametrize("db", [db])
def test_insert_and_search(db: VectorDB):
    with db.init():
        # insert
        insert_count, err = db.insert_embeddings(
            embeddings=[[1.0] * DIM] * COUNT,
            metadata=list(range(COUNT)),
        )
        assert err is None
        assert insert_count == COUNT

        db.optimize()

        # search
        results = db.search_embedding(
            query=[0.9] * DIM,
            k=10,
        )
        assert len(results) == 10
