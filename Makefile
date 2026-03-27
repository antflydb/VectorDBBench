PYTHON ?= $(CURDIR)/.venv/bin/python
BENCH = $(PYTHON) -m vectordb_bench.cli.vectordbbench
COMMON = --drop-old --skip-search-concurrent
ANTFLY_PORT ?= 18080
ANTFLY_COMMON = antflyaknn --host localhost --port $(ANTFLY_PORT) --num-shards 1
KS ?= 10 100
SEARCH_EFFORT ?=
ANTFLY_SEARCH_EFFORT_ARG = $(if $(strip $(SEARCH_EFFORT)),--search-effort $(SEARCH_EFFORT),)

help:
	@echo "VectorDBBench — Antfly Integration"
	@echo ""
	@echo "Benchmarks (50K vectors, 1536 dim):"
	@echo "  make bench-antfly-50k-low   Antfly search_effort=0.3"
	@echo "  make bench-antfly-50k-mid   Antfly search_effort=0.6"
	@echo "  make bench-antfly-50k-high  Antfly search_effort=1.0"
	@echo "  make bench-antfly-sweep     All three Antfly efforts"
	@echo "  make bench-qdrant-50k       Qdrant HNSW"
	@echo "  make bench-milvus-50k       Milvus AutoIndex"
	@echo "  make bench-chroma-50k       Chroma (isolated venv)"
	@echo "  make bench-all-50k          All DBs (50K)"
	@echo ""
	@echo "Benchmarks (1M vectors, 768 dim):"
	@echo "  make bench-antfly-1m        Antfly SPANN"
	@echo "  make bench-antfly-1m-reuse  Search-only against existing 1M Antfly table"
	@echo "  make bench-antfly-1m-compare Load once, then search at KS='$(KS)'"
	@echo "  make bench-antfly-1m-reuse-compare Search-only against existing table at KS='$(KS)'"
	@echo "  make bench-qdrant-1m        Qdrant HNSW"
	@echo "  make bench-qdrant-1m-compare Load once, then search at KS='$(KS)'"
	@echo "  make bench-milvus-1m        Milvus AutoIndex"
	@echo "  make bench-milvus-1m-compare Load once, then search at KS='$(KS)'"
	@echo "  make bench-all-1m           All three (1M)"
	@echo "  make bench-all-1m-compare   All three, each load-once + KS='$(KS)'"
	@echo "                              Optional SEARCH_EFFORT=$(SEARCH_EFFORT) for Antfly"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make start-qdrant           Start Qdrant (Docker)"
	@echo "  make start-milvus           Start Milvus (Docker)"
	@echo "  make start-chroma           Start Chroma (Docker)"
	@echo "  make stop-all               Stop Docker containers"
	@echo ""
	@echo "Dev:"
	@echo "  make unittest               Run unit tests"
	@echo "  make format / lint          Code formatting"

# --- 50K Antfly sweep ---
bench-antfly-50k-low:
	$(BENCH) $(ANTFLY_COMMON) --search-effort 0.3 \
		--case-type Performance1536D50K $(COMMON) --db-label antfly-effort-0.3

bench-antfly-50k-mid:
	$(BENCH) $(ANTFLY_COMMON) --search-effort 0.6 \
		--case-type Performance1536D50K $(COMMON) --db-label antfly-effort-0.6

bench-antfly-50k-high:
	$(BENCH) $(ANTFLY_COMMON) --search-effort 1.0 \
		--case-type Performance1536D50K $(COMMON) --db-label antfly-effort-1.0

bench-antfly-sweep: bench-antfly-50k-low bench-antfly-50k-mid bench-antfly-50k-high

# --- 50K other DBs ---
bench-qdrant-50k:
	$(BENCH) qdrantlocal --url http://localhost:6333 \
		--case-type Performance1536D50K $(COMMON) --db-label qdrant-local \
		--m 16 --ef-construct 128 --hnsw-ef 64

bench-milvus-50k:
	$(BENCH) milvusautoindex --uri http://localhost:19530 \
		--case-type Performance1536D50K $(COMMON) --db-label milvus-local

bench-chroma-50k:
	bash scripts/run_chroma_bench.sh Performance1536D50K

bench-all-50k: bench-antfly-sweep bench-qdrant-50k bench-milvus-50k bench-chroma-50k

# --- 1M benchmarks ---
bench-antfly-1m:
	$(BENCH) $(ANTFLY_COMMON) --search-effort 1.0 \
		--case-type Performance768D1M $(COMMON) --db-label antfly-local

bench-antfly-1m-reuse:
	$(BENCH) $(ANTFLY_COMMON) $(ANTFLY_SEARCH_EFFORT_ARG) \
		--case-type Performance768D1M --skip-load --skip-search-concurrent --db-label antfly-local-reuse

bench-antfly-1m-compare:
	@set -e; \
	tag=$$(date +%Y%m%d-%H%M%S); \
	$(BENCH) $(ANTFLY_COMMON) $(ANTFLY_SEARCH_EFFORT_ARG) \
		--case-type Performance768D1M --drop-old --skip-search-serial --skip-search-concurrent \
		--db-label antfly-1m-load-$$tag; \
	for k in $(KS); do \
		$(BENCH) $(ANTFLY_COMMON) $(ANTFLY_SEARCH_EFFORT_ARG) \
			--case-type Performance768D1M --skip-load --skip-search-concurrent --k $$k \
			--db-label antfly-1m-k$$k-$$tag; \
	done

bench-antfly-1m-reuse-compare:
	@set -e; \
	tag=$$(date +%Y%m%d-%H%M%S); \
	for k in $(KS); do \
		$(BENCH) $(ANTFLY_COMMON) $(ANTFLY_SEARCH_EFFORT_ARG) \
			--case-type Performance768D1M --skip-load --skip-search-concurrent --k $$k \
			--db-label antfly-1m-reuse-k$$k-$$tag; \
	done

bench-qdrant-1m:
	$(BENCH) qdrantlocal --url http://localhost:6333 \
		--case-type Performance768D1M $(COMMON) --db-label qdrant-local \
		--m 16 --ef-construct 128 --hnsw-ef 64

bench-qdrant-1m-compare:
	@set -e; \
	tag=$$(date +%Y%m%d-%H%M%S); \
	$(BENCH) qdrantlocal --url http://localhost:6333 \
		--case-type Performance768D1M --drop-old --skip-search-serial --skip-search-concurrent \
		--db-label qdrant-1m-load-$$tag --m 16 --ef-construct 128 --hnsw-ef 64; \
	for k in $(KS); do \
		$(BENCH) qdrantlocal --url http://localhost:6333 \
			--case-type Performance768D1M --skip-load --skip-search-concurrent --k $$k \
			--db-label qdrant-1m-k$$k-$$tag --m 16 --ef-construct 128 --hnsw-ef 64; \
	done

bench-milvus-1m:
	$(BENCH) milvusautoindex --uri http://localhost:19530 \
		--case-type Performance768D1M $(COMMON) --db-label milvus-local

bench-milvus-1m-compare:
	@set -e; \
	tag=$$(date +%Y%m%d-%H%M%S); \
	$(BENCH) milvusautoindex --uri http://localhost:19530 \
		--case-type Performance768D1M --drop-old --skip-search-serial --skip-search-concurrent \
		--db-label milvus-1m-load-$$tag; \
	for k in $(KS); do \
		$(BENCH) milvusautoindex --uri http://localhost:19530 \
			--case-type Performance768D1M --skip-load --skip-search-concurrent --k $$k \
			--db-label milvus-1m-k$$k-$$tag; \
	done

bench-all-1m: bench-antfly-1m bench-qdrant-1m bench-milvus-1m
bench-all-1m-compare: bench-antfly-1m-compare bench-qdrant-1m-compare bench-milvus-1m-compare

# --- Infrastructure ---
start-qdrant:
	docker run -d --name qdrant_bench -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

start-milvus:
	docker run -d --name milvus_bench -p 19530:19530 -p 9091:9091 \
		-e ETCD_USE_EMBED=true -e COMMON_STORAGETYPE=local \
		milvusdb/milvus:latest milvus run standalone

start-chroma:
	docker run -d --name chroma_bench -p 8000:8000 chromadb/chroma:latest

stop-all:
	-docker stop qdrant_bench milvus_bench chroma_bench 2>/dev/null
	-docker rm qdrant_bench milvus_bench chroma_bench 2>/dev/null

# --- Dev ---
unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench --check
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench

.PHONY: help bench-antfly-50k-low bench-antfly-50k-mid bench-antfly-50k-high bench-antfly-sweep \
	bench-qdrant-50k bench-milvus-50k bench-chroma-50k bench-all-50k \
	bench-antfly-1m bench-antfly-1m-reuse bench-antfly-1m-compare bench-antfly-1m-reuse-compare \
	bench-qdrant-1m bench-qdrant-1m-compare \
	bench-milvus-1m bench-milvus-1m-compare \
	bench-all-1m bench-all-1m-compare \
	start-qdrant start-milvus start-chroma stop-all unittest format lint
