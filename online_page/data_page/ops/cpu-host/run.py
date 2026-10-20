"""One daily CPU recovery check inside an isolated VPN container."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def vpn_command(config, credentials):
    return [
        "openconnect", "--protocol=anyconnect", "--non-inter", "--passwd-on-stdin",
        "--no-dtls", "--reconnect-timeout=30",
        "--user=" + credentials["vpn_username"],
        "--authgroup=" + config["vpnGroup"],
        "--resolve=" + config["vpnHost"] + ":" + config["vpnAddress"],
        "https://" + config["vpnHost"],
    ]


def check_once(config_file, credentials, execute):
    env = os.environ.copy()
    env["AISTATION_LOGIN_USERNAME"] = credentials["aistation_username"]
    env["AISTATION_LOGIN_PASSWORD"] = credentials["aistation_password"]
    command = ["node", "/app/aistation-cpu-keeper.mjs", "--config", str(config_file), "--refresh"]
    if execute:
        command.append("--execute")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=160)
    # Node emits only an allowlisted action/reason, never transport or login content.
    for line in (result.stdout + result.stderr).splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("action"), str):
            safe = {key: item for key, item in value.items()
                    if key in {"action", "reason", "status", "cpu", "gpu", "planned"}}
            print(json.dumps(safe), flush=True)
            return safe, result.returncode
    raise RuntimeError("KEEPER_RESULT_MISSING")


def check_with_retry(config_file, credentials, execute):
    value, code = check_once(config_file, credentials, execute)
    transient = ("HTTP_FAILED_6", "HTTP_FAILED_7", "HTTP_FAILED_28",
                 "HTTP_FAILED_52", "HTTP_FAILED_56")
    reason = value.get("reason", "")
    if code and any(reason == item or reason.startswith(item + "_") for item in transient):
        # One bounded retry within the daily job. The core persists mutation cooldowns.
        time.sleep(10)
        return check_once(config_file, credentials, execute)
    return value, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("enabled"):
        print('{"action":"DISABLED"}')
        return 0
    credentials = json.loads(args.credentials.read_text())
    log_path = Path(config["stateFile"]).parent / "vpn-private.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(log_path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    vpn = None
    try:
        with os.fdopen(descriptor, "w") as vpn_log:
            vpn = subprocess.Popen(vpn_command(config, credentials), stdin=subprocess.PIPE,
                                   stdout=vpn_log, stderr=vpn_log, text=True)
            vpn.stdin.write(credentials["vpn_password"] + "\n")
            vpn.stdin.close()
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                if vpn.poll() is not None:
                    raise RuntimeError("VPN_CONNECTION_FAILED")
                probe = subprocess.run(
                    ["curl", "--noproxy", "*", "--silent", "--fail", "--connect-timeout", "2",
                     "--max-time", "4", config["baseUrl"] + "/api/idam/v1/captcha"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if probe.returncode == 0:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("VPN_ROUTE_NOT_READY")
            value, code = check_with_retry(args.config, credentials, args.execute)
            if value["action"] == "RESTART_REQUESTED":
                time.sleep(30)
                value, code = check_once(args.config, credentials, args.execute)
            return code
    finally:
        credentials.clear()
        if vpn and vpn.poll() is None:
            vpn.send_signal(signal.SIGINT)
            try:
                vpn.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vpn.kill()
                vpn.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        reason = str(error) if isinstance(error, RuntimeError) else "HOST_CHECK_FAILED"
        print(json.dumps({"action": "NEEDS_ATTENTION", "reason": reason}), flush=True)
        raise SystemExit(1)
