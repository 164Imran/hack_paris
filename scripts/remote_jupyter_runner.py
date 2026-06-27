from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid

import websocket


class RemoteJupyterRunner:
    """Execute Python code on a remote Jupyter server through its REST/WebSocket API."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.token = token
        self.kernel_id: str | None = None
        self.ws: websocket.WebSocket | None = None

    def start(self) -> None:
        kernel = self.request("/api/kernels", method="POST", body={"name": "python3"})
        self.kernel_id = kernel["id"]
        ws_url = f"{self.ws_base_url}/api/kernels/{self.kernel_id}/channels?" + urllib.parse.urlencode(
            {"token": self.token}
        )
        self.ws = websocket.create_connection(ws_url, timeout=30)

    def stop(self) -> None:
        if self.ws is not None:
            self.ws.close()
        if self.kernel_id is not None:
            self.request_no_json(f"/api/kernels/{self.kernel_id}", method="DELETE")

    def execute(self, code: str, timeout: int = 3600) -> str:
        if self.ws is None:
            raise RuntimeError("Runner is not started")
        msg_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": msg_id,
                "username": "codex",
                "session": uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }
        self.ws.send(json.dumps(message))
        output: list[str] = []
        started = time.time()
        while time.time() - started < timeout:
            raw = self.ws.recv()
            message = json.loads(raw)
            if message.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = message["header"]["msg_type"]
            content = message["content"]
            if msg_type == "stream":
                output.append(content.get("text", ""))
            elif msg_type in {"execute_result", "display_data"}:
                output.append(content.get("data", {}).get("text/plain", ""))
            elif msg_type == "error":
                output.append("\n".join(content.get("traceback", [])))
                break
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return "".join(output)

    def request(self, path: str, method: str = "GET", body: dict | None = None) -> dict:
        with urllib.request.urlopen(self.make_request(path, method, body), timeout=30) as response:
            return json.load(response)

    def request_no_json(self, path: str, method: str = "GET", body: dict | None = None) -> None:
        with urllib.request.urlopen(self.make_request(path, method, body), timeout=30):
            return None

    def make_request(self, path: str, method: str, body: dict | None) -> urllib.request.Request:
        url = self.base_url + path + "?" + urllib.parse.urlencode({"token": self.token})
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Python file on a remote Jupyter kernel.")
    parser.add_argument("python_file", help="Local Python file whose contents will be executed remotely.")
    parser.add_argument("--prelude", action="append", default=[], help="Local Python file to execute first in the same kernel.")
    parser.add_argument(
        "--upload",
        action="append",
        default=[],
        help="Upload a local file before execution, formatted as local_path:remote_path.",
    )
    parser.add_argument(
        "--download",
        action="append",
        default=[],
        help="Download a remote file after execution, formatted as remote_path:local_path.",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def upload_file(runner: RemoteJupyterRunner, spec: str, timeout: int) -> None:
    local_value, remote_value = spec.split(":", maxsplit=1)
    local_path = Path(local_value)
    remote_path = remote_value
    chunk_size = int(os.environ.get("JUPYTER_UPLOAD_CHUNK_BYTES", str(8 * 1024 * 1024)))
    file_size = local_path.stat().st_size
    resume = os.environ.get("JUPYTER_UPLOAD_RESUME") == "1"
    uploaded = 0
    if resume:
        code = f"""
from pathlib import Path
target = Path({remote_path!r})
target.parent.mkdir(parents=True, exist_ok=True)
print("remote_size", target.stat().st_size if target.exists() else 0)
"""
        output = runner.execute(code, timeout=timeout)
        print(output)
        match = re.search(r"remote_size\s+(\d+)", output)
        uploaded = int(match.group(1)) if match else 0
        uploaded = min(uploaded, file_size)
    if uploaded == 0:
        code = f"""
from pathlib import Path
target = Path({remote_path!r})
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(b"")
print("initialized", target)
"""
        print(runner.execute(code, timeout=timeout))
    else:
        print(f"resuming {local_path} at {uploaded}/{file_size} bytes")
    chunk_index = uploaded // chunk_size
    with local_path.open("rb") as f:
        f.seek(uploaded)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_index += 1
            uploaded += len(chunk)
            payload = base64.b64encode(chunk).decode("ascii")
            code = f"""
from pathlib import Path
import base64
target = Path({remote_path!r})
with target.open("ab") as f:
    f.write(base64.b64decode({payload!r}))
"""
            if chunk_index == 1 or chunk_index % 25 == 0 or uploaded == file_size:
                print(f"uploading {local_path} chunk {chunk_index} ({uploaded}/{file_size} bytes)")
            output = runner.execute(code, timeout=timeout)
            if output.strip() and (chunk_index == 1 or chunk_index % 25 == 0 or uploaded == file_size):
                print(output)
    code = f"""
from pathlib import Path
target = Path({remote_path!r})
actual = target.stat().st_size
expected = {file_size}
print("uploaded", target, actual, "bytes")
assert actual == expected, (actual, expected)
"""
    print(runner.execute(code, timeout=timeout))


def download_file(runner: RemoteJupyterRunner, spec: str, timeout: int) -> None:
    remote_value, local_value = spec.split(":", maxsplit=1)
    local_path = Path(local_value)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = int(os.environ.get("JUPYTER_DOWNLOAD_CHUNK_BYTES", str(1024 * 1024)))
    code = f"""
from pathlib import Path
target = Path({remote_value!r})
print("remote_size", target.stat().st_size)
"""
    output = runner.execute(code, timeout=timeout)
    print(output)
    match = re.search(r"remote_size\s+(\d+)", output)
    if not match:
        raise RuntimeError(f"Could not determine remote size for {remote_value}")
    file_size = int(match.group(1))
    with local_path.open("wb") as f:
        for offset in range(0, file_size, chunk_size):
            code = f"""
from pathlib import Path
import base64
target = Path({remote_value!r})
with target.open("rb") as f:
    f.seek({offset})
    data = f.read({chunk_size})
print("__BEGIN_CHUNK__")
print(base64.b64encode(data).decode("ascii"))
print("__END_CHUNK__")
"""
            output = runner.execute(code, timeout=timeout)
            match = re.search(r"__BEGIN_CHUNK__\n(.*?)\n__END_CHUNK__", output, flags=re.S)
            if not match:
                raise RuntimeError(f"Could not parse download chunk at offset {offset}")
            f.write(base64.b64decode(match.group(1)))
            if offset == 0 or offset + chunk_size >= file_size:
                print(f"downloaded {local_path} {min(offset + chunk_size, file_size)}/{file_size} bytes")
    actual = local_path.stat().st_size
    if actual != file_size:
        raise RuntimeError(f"Downloaded size mismatch for {local_path}: {actual} != {file_size}")


def main() -> None:
    args = parse_args()
    base_url = os.environ["JUPYTER_URL"].split("/lab?")[0].rstrip("/")
    token = os.environ["JUPYTER_TOKEN"]
    runner = RemoteJupyterRunner(base_url=base_url, token=token)
    try:
        runner.start()
        for upload in args.upload:
            upload_file(runner, upload, timeout=args.timeout)
        for prelude in args.prelude:
            print(runner.execute(open(prelude, encoding="utf-8").read(), timeout=args.timeout))
        code = open(args.python_file, encoding="utf-8").read()
        print(runner.execute(code, timeout=args.timeout))
        for download in args.download:
            download_file(runner, download, timeout=args.timeout)
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
