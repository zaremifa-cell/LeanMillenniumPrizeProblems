import json
import subprocess
import threading
from typing import Any, Dict, List, Optional


class LeanOracle:
    def __init__(self, lean_file: str = "autoprover/lean/Oracle.lean") -> None:
        cmd = ["lake", "env", "lean", "--run", lean_file]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if not self.proc.stderr:
            return
        for line in self.proc.stderr:
            self._stderr_lines.append(line)

    def request(
        self,
        goal_expr: str,
        local_context: List[Dict[str, str]],
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("Lean oracle process not started")
        req = {
            "goal_expr": goal_expr,
            "local_context": local_context,
            "action": action,
        }
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        # Read until we get a non-empty JSON line or the process exits
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err_tail = self.stderr_tail(50)
                raise RuntimeError(
                    "No response from Lean oracle. stderr tail:\n" + err_tail
                )
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Keep reading if we got a non-JSON line
                continue

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.terminate()

    def stderr_tail(self, n: int = 20) -> str:
        return "".join(self._stderr_lines[-n:])

