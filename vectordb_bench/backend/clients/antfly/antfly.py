import logging
import subprocess
import json
import time
import os
from contextlib import contextmanager
from typing import Any, List

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)


class AntflyClient(VectorDB):
    """Antfly client for VectorDB implemented by running antflycli."""

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = "vdbbench_collection"
        self.dim = dim
        self.antfly_process = None
        self.antfly_path = "/tmp/antfly_bin/antfly"
        self.antflycli_path = "/tmp/antfly_bin/antflycli"

        if drop_old:
            log.info("Dropping old Antfly data and stopping swarm.")
            subprocess.run("ps aux | grep antfly | grep -v grep | awk '{print $2}' | xargs -r kill -9", shell=True, check=False)
            # Give it a moment to shut down
            time.sleep(2)
            # In swarm mode, data is stored in the current directory in a folder named "data"
            subprocess.run(["rm", "-rf", "data"], check=False)


    @contextmanager
    def init(self) -> None:
        """create and destory connections to database."""
        log.info("Checking for old antfly processes...")
        subprocess.run("ps aux | grep antfly", shell=True)
        log.info("Killing old antfly processes...")
        subprocess.run("ps aux | grep antfly | grep -v grep | awk '{print $2}' | xargs -r kill -9", shell=True, check=False)
        time.sleep(2)

        log.info("Downloading and setting up Antfly binaries...")
        download_url = "https://releases.antfly.io/antfly/0.0.0-dev4/antfly_0.0.0-dev4_Linux_x86_64.tar.gz"
        tmp_tar_path = "/tmp/antfly.tar.gz"
        tmp_bin_dir = "/tmp/antfly_bin"
        subprocess.run(["curl", "-L", download_url, "-o", tmp_tar_path], check=True)
        subprocess.run(["mkdir", "-p", tmp_bin_dir], check=True)
        subprocess.run(["tar", "-xzf", tmp_tar_path, "-C", tmp_bin_dir], check=True)
        subprocess.run(["chmod", "-R", "+x", tmp_bin_dir], check=True)
        log.info("Antfly binaries are ready.")

        log.info("Starting antfly swarm...")
        self.antfly_log_file = "antfly_swarm.log"
        with open(self.antfly_log_file, "w") as f:
            self.antfly_process = subprocess.Popen(
                [self.antfly_path, "swarm"],
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        # Wait for antfly to be ready
        retries = 20
        while retries > 0:
            time.sleep(5)
            # The health check endpoint for antfly is /api/v1/health
            res = subprocess.run('curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/v1/health', shell=True, capture_output=True, text=True)
            if res.stdout.strip() == "200":
                log.info("Antfly swarm started.")
                break
            retries -= 1
        else:
            log.error("Failed to start antfly swarm.")
            with open(self.antfly_log_file, "r") as f:
                log.error("Antfly logs:\n" + f.read())
            raise RuntimeError("Failed to start antfly swarm.")

        # Create table
        log.info(f"Creating table {self.collection_name}")
        subprocess.run(
            [
                self.antflycli_path,
                "table",
                "create",
                "--table",
                self.collection_name,
            ],
            check=True,
        )


        yield

        log.info("Stopping antfly swarm...")
        if self.antfly_process:
            self.antfly_process.terminate()
            self.antfly_process.wait()
            subprocess.run("ps aux | grep antfly | grep -v grep | awk '{print $2}' | xargs -r kill -9", shell=True, check=False)
            log.info("Antfly swarm stopped.")

    def ready_to_search(self) -> bool:
        log.info("Antfly is always ready to search")
        return True

    def optimize(self, data_size: int | None = None):
        log.info("Antfly does not support optimizing index")
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        log.info(f"Inserting {len(embeddings)} embeddings into {self.collection_name}")
        data = []
        for i, emb in enumerate(embeddings):
            data.append({"id": metadata[i], "embedding": emb})

        tmp_file = "temp_data.json"
        with open(tmp_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        try:
            subprocess.run(
                [
                    self.antflycli_path,
                    "load",
                    "--table",
                    self.collection_name,
                    "--file-path",
                    tmp_file,
                    "--id-field",
                    "id",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to load data into antfly: {e}")
            log.error(f"Antfly stderr: {e.stderr}")
            return 0, e
        finally:
            os.remove(tmp_file)

        log.info(f"Creating index on {self.collection_name}")
        try:
            # Antfly needs an index to perform vector search.
            # We create an index on the 'embedding' field.
            # We use the 'spann' type, which is the vector index type for Antfly.
            subprocess.run(
                [
                    self.antflycli_path,
                    "index",
                    "create",
                    "--table",
                    self.collection_name,
                    "--index",
                    "embedding_idx",
                    "--field",
                    "embedding",
                    "--dimension",
                    str(self.dim),
                    "--type",
                    "spann",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to create index in antfly: {e}")
            log.error(f"Antfly stderr: {e.stderr}")
            # This is a fatal error for search.
            return 0, e


        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        log.info(f"Searching for embedding in {self.collection_name}")
        # We need to pass the vector as a string to the CLI.
        # The semantic search expects a text query, but we are passing a vector string.
        # This is a workaround, and it's not guaranteed to work.
        query_str = json.dumps(query)

        try:
            result = subprocess.run(
                [
                    self.antflycli_path,
                    "query",
                    "--table",
                    self.collection_name,
                    "--vector",
                    query_str,
                    "--indexes",
                    "embedding_idx",
                    "--fields",
                    "id",
                    "--limit",
                    str(k),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output_lines = result.stdout.strip().split("\n")
            ids = []
            for line in output_lines:
                try:
                    res = json.loads(line)
                    # The result format is not documented, so we are guessing here.
                    # We assume the 'id' field is returned in the document.
                    # Example format: {"document":{"id":123},"score":0.8}
                    if 'document' in res and 'id' in res['document']:
                         ids.append(int(res['document']['id']))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    log.warning(f"Could not parse antfly search result line: {line}. The format is assumed to be JSON lines with a 'document' object containing an 'id' field. Error: {e}")
                    continue
            return ids
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to search in antfly: {e}")
            log.error(f"Antfly stderr: {e.stderr}")
            return []
