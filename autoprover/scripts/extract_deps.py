import argparse
import json
import subprocess
from typing import Dict, Iterable


def iter_lines(cmd: list[str]) -> Iterable[Dict]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
    proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-prefix", default="Mathlib.Data")
    ap.add_argument("--kind", default="theorem")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="autoprover/data/decls.jsonl")
    args = ap.parse_args()

    cmd = [
        "lake",
        "env",
        "lean",
        "--run",
        "autoprover/lean/ExtractDeps.lean",
        "--",
        "--module-prefix",
        args.module_prefix,
    ]
    if args.kind:
        cmd += ["--kind", args.kind]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for obj in iter_lines(cmd):
            if args.kind and obj.get("kind") != args.kind:
                continue
            f.write(json.dumps(obj) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} decls to {args.out}")


if __name__ == "__main__":
    main()
