import argparse
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoprover.utils.decls import load_decls
from autoprover.utils.lean_oracle import LeanOracle
from autoprover.utils.tokenize import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decls", default="autoprover/data/decls.jsonl")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    decls = load_decls(args.decls)
    oracle = LeanOracle()

    success = 0
    total = 0
    for d in decls[: args.limit]:
        goal = d.get("type_expr", "")
        if not goal:
            continue
        total += 1
        state_goals = [goal]
        ctx = []
        for _ in range(args.max_steps):
            if not state_goals:
                success += 1
                break
            # simple baseline: random apply on a random theorem
            lemma = random.choice(decls)["decl_name"]
            action = {"kind": "apply", "lemma": lemma}
            resp = oracle.request(state_goals[0], ctx, action)
            if not resp.get("ok", False):
                continue
            state_goals = resp.get("goals", [])
            ctx = resp.get("context", [])
        if not state_goals:
            success += 1

    oracle.close()
    if total == 0:
        print("No valid goals found")
    else:
        print(f"Success rate: {success}/{total} = {success/total:.2%}")


if __name__ == "__main__":
    main()
