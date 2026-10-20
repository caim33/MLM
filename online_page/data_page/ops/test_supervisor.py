"""Linux integration checks using disposable HTTP children, no real datasets."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform == "linux", "requires Linux process groups and flock")
class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="viewer-supervisor-test-")
        self.root = Path(self.directory.name)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "runtime_root": str(self.root),
            "interval_seconds": 0.2,
            "services": {"test": {
                "argv": [sys.executable, "-m", "http.server", str(port),
                         "--bind", "127.0.0.1", "--directory", str(self.root)],
                "health_url": f"http://127.0.0.1:{port}/",
                "startup_grace": 1, "failure_limit": 1,
            }},
        }))
        self.command = [sys.executable, str(Path(__file__).with_name("viewer-supervisor.py")),
                        "--config", str(self.config)]
        self.supervisor_pid = None

    def wait_for(self, predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = json.loads((self.root / "supervisor.json").read_text())
                self.supervisor_pid = state["pid"]
                if predicate(state):
                    return state
            except (OSError, KeyError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        self.fail("Supervisor did not reach expected state")

    def start(self):
        subprocess.run(self.command + ["start"], check=True)
        return self.wait_for(lambda s: s["services"]["test"]["healthy"])

    def tearDown(self):
        if self.supervisor_pid:
            try:
                os.kill(self.supervisor_pid, signal.SIGTERM)
                self.wait_for(lambda s: s["status"] == "stopped")
            except ProcessLookupError:
                pass
        self.directory.cleanup()

    def test_duplicate_start_and_crash_recovery(self):
        initial = self.start()
        launches = [subprocess.Popen(self.command + ["start"]) for _ in range(5)]
        for launch in launches:
            self.assertEqual(launch.wait(timeout=10), 0)
        unchanged = json.loads((self.root / "supervisor.json").read_text())
        self.assertEqual(unchanged["pid"], initial["pid"])
        old_pid = initial["services"]["test"]["pid"]
        self.assertEqual(unchanged["services"]["test"]["pid"], old_pid)
        os.kill(old_pid, signal.SIGKILL)
        restored = self.wait_for(lambda s: s["services"]["test"]["healthy"]
                                and s["services"]["test"]["pid"] != old_pid)
        self.assertEqual(restored["services"]["test"]["starts"], 2)

    def test_hung_child_is_replaced(self):
        initial = self.start()
        old_pid = initial["services"]["test"]["pid"]
        os.kill(old_pid, signal.SIGSTOP)
        restored = self.wait_for(lambda s: s["services"]["test"]["healthy"]
                                and s["services"]["test"]["pid"] != old_pid, timeout=35)
        self.assertEqual(restored["services"]["test"]["starts"], 2)


if __name__ == "__main__":
    unittest.main()
