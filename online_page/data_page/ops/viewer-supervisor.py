#!/usr/bin/env python3
"""Supervise the viewer and its tunnel. Linux, Python standard library only."""
import argparse
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener


HTTP = build_opener(ProxyHandler({}))


def healthy(url):
    try:
        with HTTP.open(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def terminate(child):
    # Only signal an unreaped child owned by this supervisor.
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def supervise(config, root, lock):
    log = logging.getLogger("viewer-service")
    log.setLevel(logging.INFO)
    handler = RotatingFileHandler(root / "logs" / "supervisor.log",
                                  maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    services = {}
    for name, spec in config["services"].items():
        services[name] = dict(spec=spec, child=None, starts=0, failures=0,
                              backoff=2, next_start=0, started=0, healthy=False)
    atomic_json(root / "supervisor.json", {"pid": os.getpid(), "status": "starting"})
    log.info("supervisor started")
    try:
        while not stopping:
            now = time.monotonic()
            for name, service in services.items():
                spec, child = service["spec"], service["child"]
                if child is not None and child.poll() is not None:
                    log.warning("%s exited code=%s", name, child.returncode)
                    service["child"] = child = None
                    service["healthy"] = False
                    service["next_start"] = now + service["backoff"]
                    service["backoff"] = min(60, service["backoff"] * 2)
                if child is None and now >= service["next_start"]:
                    logfile = root / "logs" / (name + ".log")
                    if logfile.exists() and logfile.stat().st_size > 10_000_000:
                        logfile.replace(logfile.with_suffix(".log.previous"))
                    try:
                        with logfile.open("ab") as output:
                            child = subprocess.Popen(
                                spec["argv"], cwd=spec.get("cwd"), stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT,
                                env={**os.environ, **config.get("env", {})},
                                start_new_session=True, close_fds=True)
                        service.update(child=child, started=now, failures=0,
                                       starts=service["starts"] + 1)
                        (root / (name + ".pid")).write_text(str(child.pid) + "\n")
                        log.info("%s started pid=%s", name, child.pid)
                    except OSError as error:
                        log.error("%s start failed type=%s", name, type(error).__name__)
                        service["next_start"] = now + service["backoff"]
                        service["backoff"] = min(60, service["backoff"] * 2)
                if child is not None and child.poll() is None:
                    service["healthy"] = healthy(spec["health_url"])
                    if service["healthy"]:
                        service["failures"] = 0
                        if now - service["started"] >= 60:
                            service["backoff"] = 2
                    elif now - service["started"] > spec.get("startup_grace", 60):
                        service["failures"] += 1
                        if service["failures"] >= spec.get("failure_limit", 3):
                            log.warning("%s failed health checks; restarting owned child", name)
                            terminate(child)
            atomic_json(root / "supervisor.json", {
                "pid": os.getpid(), "status": "running", "updated": time.time(),
                "services": {name: {
                    "pid": s["child"].pid if s["child"] else None,
                    "healthy": s["healthy"], "starts": s["starts"],
                    "failures": s["failures"],
                } for name, s in services.items()},
            })
            time.sleep(config.get("interval_seconds", 5))
    finally:
        for service in services.values():
            terminate(service["child"])
        atomic_json(root / "supervisor.json", {"pid": os.getpid(), "status": "stopped"})
        log.info("supervisor stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("action", choices=["start", "run", "status"])
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    root = Path(config["runtime_root"])
    if args.action == "status":
        state = json.loads((root / "supervisor.json").read_text())
        state["checks"] = {name: healthy(s["health_url"])
                           for name, s in config["services"].items()}
        state["fresh"] = time.time() - state.get("updated", 0) < 30
        print(json.dumps(state, indent=2))
        return 0 if state["fresh"] and all(state["checks"].values()) else 1
    (root / "logs").mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if args.action == "start":
            # Transfer the already-held lock to the child: concurrent launches
            # cannot start duplicate processes or race on the status file.
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 "--config", str(config_path), "run"],
                env={**os.environ, "VIEWER_LOCK_FD": str(lock.fileno())},
                pass_fds=(lock.fileno(),), start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return 0
        supervise(config, root, lock)
    return 0


if __name__ == "__main__":
    # A start command transfers its lock; direct run acquires it itself.
    inherited = os.environ.pop("VIEWER_LOCK_FD", None)
    if inherited is not None:
        lock_fd = int(inherited)
        os.set_inheritable(lock_fd, False)
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, type=Path)
        parser.add_argument("action")
        arguments = parser.parse_args()
        settings = json.loads(arguments.config.read_text())
        with os.fdopen(lock_fd, "a") as transferred_lock:
            supervise(settings, Path(settings["runtime_root"]), transferred_lock)
    else:
        sys.exit(main())
