import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoprover.kg.build_graph import build_graph, load_decls, save_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="autoprover/data/decls.jsonl")
    ap.add_argument("--out", default="autoprover/data/kg.graph.json")
    args = ap.parse_args()

    decls = load_decls(args.inp)
    graph = build_graph(decls)
    save_graph(graph, args.out)
    print(f"Wrote graph to {args.out} (nodes={len(graph['nodes'])})")


if __name__ == "__main__":
    main()
