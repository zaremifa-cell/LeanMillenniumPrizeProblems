import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoprover.kg.pretrain_embeddings import load_graph, pretrain_embeddings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="autoprover/data/kg.graph.json")
    ap.add_argument("--out", default="autoprover/data/kg.emb.pt")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=5)
    args = ap.parse_args()

    graph = load_graph(args.graph)
    pretrain_embeddings(graph, args.out, dim=args.dim, epochs=args.epochs, negatives=args.negatives)
    print(f"Saved embeddings to {args.out}")


if __name__ == "__main__":
    main()
