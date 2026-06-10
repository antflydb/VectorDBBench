import logging
import time

import pytest
import ujson

from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import (
    DB,
    CaseConfig,
    CaseType,
    IndexType,
    TaskConfig,
)

log = logging.getLogger(__name__)

class TestBenchRunner:
    def test_get_results(self):
        runner = BenchMarkRunner()

        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_whole(self):
        pytest.skip("requires a live Milvus server")
        runner = BenchMarkRunner()

        task_config=TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        runner._sync_running_task()
        result = runner.get_results()
        log.info(f"test result: {result}")

    def test_performance_case_clean(self):
        pytest.skip("requires a live Milvus server")
        runner = BenchMarkRunner()

        task_config=TaskConfig(
            db=DB.Milvus,
            db_config=DB.Milvus.config(),
            db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
            case_config=CaseConfig(case_id=CaseType.PerformanceSZero),
        )

        runner.run([task_config])
        time.sleep(3)
        runner.stop_running()

    def test_performance_case_no_error(self):
        task_config=TaskConfig(
            db=DB.ZillizCloud,
            db_config=DB.ZillizCloud.config_cls(uri="xxx", user="abc", password="1234"),  # noqa: S106
            db_case_config=DB.ZillizCloud.case_config_cls()(),
            case_config=CaseConfig(case_id=CaseType.Performance1536D50K),
        )

        t = task_config.copy()
        d = t.model_dump_json(exclude={'db_config': {'password', 'api_key'}})
        log.info(f"{d}")

        loads = ujson.loads(d)
        log.info(f"{loads}")
