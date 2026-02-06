import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoprover.muzero.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decls", default="autoprover/data/decls.jsonl")
    ap.add_argument("--graph", default="autoprover/data/kg.graph.json")
    ap.add_argument("--emb", default="autoprover/data/kg.emb.pt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    train(args.decls, args.graph, args.emb, steps=args.steps, batch_size=args.batch)


if __name__ == "__main__":
    main()
