lint:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench --check
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench

unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

.PHONY: lint unittest
