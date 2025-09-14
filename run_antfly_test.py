import traceback
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.backend.clients import DB, EmptyDBCaseConfig
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.models import TaskConfig, CaseConfig
from vectordb_bench.backend.clients.antfly.config import AntflyConfig

def run_test():
    print("Starting Antfly test...")
    try:
        taskLabel = "antfly_test"
        tasks = [
            TaskConfig(
                db=DB.Antfly,
                db_config=AntflyConfig(),
                db_case_config=EmptyDBCaseConfig(),
                case_config=CaseConfig(case_id=CaseType.Performance1536D50K),
            )
        ]
        benchmark_runner.set_drop_old(True)
        print("Running benchmark...")
        benchmark_runner.run(tasks, taskLabel)
        print("Benchmark finished.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
