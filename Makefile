PYTHON ?= /home/rowan/.local/share/uv/tools/pip/bin/python
BENCH = $(PYTHON) -m vectordb_bench.cli.vectordbbench
COMMON = --drop-old --skip-search-concurrent

help:
	@echo "VectorDBBench — Antfly Integration"
	@echo ""
	@echo "Benchmarks (50K vectors, 1536 dim):"
	@echo "  make bench-antfly-50k    Antfly SPANN"
	@echo "  make bench-qdrant-50k    Qdrant HNSW"
	@echo "  make bench-milvus-50k    Milvus AutoIndex"
	@echo "  make bench-all-50k       All three (50K)"
	@echo ""
	@echo "Benchmarks (1M vectors, 768 dim):"
	@echo "  make bench-antfly-1m     Antfly SPANN"
	@echo "  make bench-qdrant-1m     Qdrant HNSW"
	@echo "  make bench-milvus-1m     Milvus AutoIndex"
	@echo "  make bench-all-1m        All three (1M)"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make start-qdrant        Start Qdrant (Docker)"
	@echo "  make start-milvus        Start Milvus (Docker)"
	@echo "  make stop-all            Stop Docker containers"
	@echo ""
	@echo "Dev:"
	@echo "  make unittest            Run unit tests"
	@echo "  make format / lint       Code formatting"

# --- 50K benchmarks ---
bench-antfly-50k:
	$(BENCH) antflyaknn --host localhost --port 8080 --num-shards 1 \
		--case-type Performance1536D50K $(COMMON) --db-label antfly-local

bench-qdrant-50k:
	$(BENCH) qdrantlocal --url http://localhost:6333 \
		--case-type Performance1536D50K $(COMMON) --db-label qdrant-local \
		--m 16 --ef-construct 128 --hnsw-ef 64

bench-milvus-50k:
	$(BENCH) milvusautoindex --uri http://localhost:19530 \
		--case-type Performance1536D50K $(COMMON) --db-label milvus-local

bench-all-50k: bench-antfly-50k bench-qdrant-50k bench-milvus-50k

# --- 1M benchmarks ---
bench-antfly-1m:
	$(BENCH) antflyaknn --host localhost --port 8080 --num-shards 1 \
		--case-type Performance768D1M $(COMMON) --db-label antfly-local

bench-qdrant-1m:
	$(BENCH) qdrantlocal --url http://localhost:6333 \
		--case-type Performance768D1M $(COMMON) --db-label qdrant-local \
		--m 16 --ef-construct 128 --hnsw-ef 64

bench-milvus-1m:
	$(BENCH) milvusautoindex --uri http://localhost:19530 \
		--case-type Performance768D1M $(COMMON) --db-label milvus-local

bench-all-1m: bench-antfly-1m bench-qdrant-1m bench-milvus-1m

# --- Infrastructure ---
start-qdrant:
	docker run -d --name qdrant_bench -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

start-milvus:
	docker run -d --name milvus_bench -p 19530:19530 -p 9091:9091 \
		-e ETCD_USE_EMBED=true -e COMMON_STORAGETYPE=local \
		milvusdb/milvus:latest milvus run standalone

stop-all:
	-docker stop qdrant_bench milvus_bench 2>/dev/null
	-docker rm qdrant_bench milvus_bench 2>/dev/null

# --- Dev ---
unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench --check
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench

.PHONY: help bench-antfly-50k bench-qdrant-50k bench-milvus-50k bench-all-50k \
	bench-antfly-1m bench-qdrant-1m bench-milvus-1m bench-all-1m \
	start-qdrant start-milvus stop-all unittest format lint
