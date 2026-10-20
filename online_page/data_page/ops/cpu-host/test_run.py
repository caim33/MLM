import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("host_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class HostRunnerTest(unittest.TestCase):
    def test_vpn_password_never_appears_in_arguments(self):
        args = runner.vpn_command({"vpnHost": "vpn.example", "vpnAddress": "127.0.0.1", "vpnGroup": "staff"},
                                  {"vpn_username": "account", "vpn_password": "secret"})
        self.assertNotIn("secret", " ".join(args))
        self.assertIn("--passwd-on-stdin", args)

    def test_platform_credentials_use_environment_and_no_exercise(self):
        credentials = {"aistation_username": "account", "aistation_password": "secret"}
        response = subprocess.CompletedProcess([], 0, '{"action":"HEALTHY"}\n', "")
        with patch.object(runner.subprocess, "run", return_value=response) as call:
            value, code = runner.check_once(Path("config.json"), credentials, True)
            args, kwargs = call.call_args
            self.assertNotIn("secret", " ".join(args[0]))
            self.assertEqual(kwargs["env"]["AISTATION_LOGIN_PASSWORD"], "secret")
            self.assertIn("--execute", args[0])
            self.assertNotIn("--exercise-restart", args[0])
            self.assertEqual((value["action"], code), ("HEALTHY", 0))

    def test_dry_run_and_unknown_result(self):
        credentials = {"aistation_username": "account", "aistation_password": "secret"}
        response = subprocess.CompletedProcess([], 1, "", "invalid")
        with patch.object(runner.subprocess, "run", return_value=response) as call:
            with self.assertRaisesRegex(RuntimeError, "KEEPER_RESULT_MISSING"):
                runner.check_once(Path("config.json"), credentials, False)
            self.assertNotIn("--execute", call.call_args.args[0])

    def test_only_transient_network_failure_gets_one_retry(self):
        failure = ({"action": "NEEDS_ATTENTION", "reason": "HTTP_FAILED_28_LOGIN"}, 1)
        success = ({"action": "HEALTHY"}, 0)
        with patch.object(runner, "check_once", side_effect=[failure, success]) as call, \
                patch.object(runner.time, "sleep"):
            self.assertEqual(runner.check_with_retry(Path("config.json"), {}, True), success)
            self.assertEqual(call.call_count, 2)
        denied = ({"action": "NEEDS_ATTENTION", "reason": "API_LOGIN_REJECTED"}, 1)
        with patch.object(runner, "check_once", return_value=denied) as call:
            self.assertEqual(runner.check_with_retry(Path("config.json"), {}, True), denied)
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
