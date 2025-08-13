import logging
import subprocess
from pathlib import Path
import os
import stat
import time
import httpx

import click

log = logging.getLogger(__name__)

ANTFLY_DIR = Path.home() / ".vectordb_bench" / "antfly"
PID_FILE = ANTFLY_DIR / "antfly.pid"
ANTFLY_URL = "https://releases.antfly.io/antfly_0.0.0-dev1-SNAPSHOT-f561585_Linux_x86_64.tar.gz"
ANTFLY_ARCHIVE = ANTFLY_DIR / "antfly_linux_x86_64.tar.gz"
ANTFLY_BINARY = ANTFLY_DIR / "antfly"

def run_cmd(cmd, cwd=None, check=True):
    log.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=check)

@click.group()
def antfly():
    """Antfly local commands"""
    pass

@antfly.command()
@click.option("--url", default=ANTFLY_URL, help="URL of the Antfly binary to download.")
def up(url):
    """Start an Antfly local instance."""
    ANTFLY_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTFLY_BINARY.exists():
        log.info(f"Downloading Antfly from {url}...")
        run_cmd(["wget", "-O", str(ANTFLY_ARCHIVE), url])
        log.info("Extracting Antfly...")
        run_cmd(["tar", "-xzf", str(ANTFLY_ARCHIVE)], cwd=str(ANTFLY_DIR))

        antfly_cli_binary = ANTFLY_DIR / "antflycli"
        os.chmod(ANTFLY_BINARY, os.stat(ANTFLY_BINARY).st_mode | stat.S_IEXEC)
        os.chmod(antfly_cli_binary, os.stat(antfly_cli_binary).st_mode | stat.S_IEXEC)

    log.info("Starting Antfly swarm...")
    proc = subprocess.Popen([str(ANTFLY_BINARY), "swarm"], cwd=str(ANTFLY_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    log.info(f"Antfly started with PID {proc.pid}")

    log.info("Waiting for Antfly to start...")
    max_wait = 30
    wait_interval = 1
    elapsed = 0

    while elapsed < max_wait:
        try:
            res = httpx.get("http://localhost:8080/table")
            if res.status_code == 200:
                log.info("Antfly is up and running.")
                return
        except httpx.RequestError:
            pass
        time.sleep(wait_interval)
        elapsed += wait_interval

    log.error(f"Antfly failed to start within {max_wait} seconds.")
    down.callback() # try to clean up
    raise RuntimeError("Antfly failed to start")


@antfly.command()
def down():
    """Stop an Antfly local instance."""
    if not PID_FILE.exists():
        log.info("Antfly PID file not found. Server may not be running.")
        return

    with open(PID_FILE, "r") as f:
        pid_str = f.read()
        if not pid_str:
            log.info("PID file is empty.")
            return
        pid = int(pid_str)

    log.info(f"Stopping Antfly with PID {pid}...")
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        log.info(f"Antfly with PID {pid} stopped.")
    except ProcessLookupError:
        log.info(f"Process with PID {pid} not found. It might have already been stopped.")
    except Exception as e:
        log.error(f"Failed to stop Antfly with PID {pid}: {e}")

    PID_FILE.unlink()
