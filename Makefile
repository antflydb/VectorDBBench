install:
	pip install -e .

install-all:
	pip install -e '.[all]'

install-dev:
	pip install -e '.[test]'

run:
	python -m vectordb_bench

# uvx targets for Arch Linux (externally-managed environment)
uvx-run:
	uvx --python 3.11 --from . init_bench

uvx-run-all:
	uvx --python 3.11 --from . --with '.[all]' init_bench

uvx-cli:
	uvx --python 3.11 --from . --with '.[all]' vectordbbench

unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench --check
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench
