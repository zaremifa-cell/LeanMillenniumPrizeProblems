import json
from collections import defaultdict
from typing import Dict, Iterable, List, Set


def load_decls(path: str) -> List[Dict]:
    decls: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            decls.append(json.loads(line))
    return decls


def build_graph(decls: List[Dict]) -> Dict:
    nodes: Dict[str, Dict] = {}
    edges: Dict[str, Set[str]] = defaultdict(set)

    for d in decls:
        name = d["decl_name"]
        kind = d.get("kind", "unknown")
        nodes[name] = {
            "kind": kind,
            "module": d.get("module", ""),
        }
        deps = set(d.get("type_consts", [])) | set(d.get("value_consts", []))
        deps.discard(name)
        edges[name].update(deps)

    # add external nodes referenced but not declared
    for src, deps in list(edges.items()):
        for dep in deps:
            if dep not in nodes:
                nodes[dep] = {"kind": "external", "module": ""}

    # compute degree
    for n in nodes:
        nodes[n]["degree"] = len(edges.get(n, []))

    # compute depth (longest path to leaf)
    memo: Dict[str, int] = {}
    visiting: Set[str] = set()

    def depth(n: str) -> int:
        if n in memo:
            return memo[n]
        if n in visiting:
            return 0
        visiting.add(n)
        deps = edges.get(n, [])
        if not deps:
            memo[n] = 0
        else:
            memo[n] = 1 + max(depth(d) for d in deps)
        visiting.remove(n)
        return memo[n]

    for n in nodes:
        nodes[n]["depth"] = depth(n)

    return {"nodes": nodes, "edges": {k: sorted(list(v)) for k, v in edges.items()}}


def save_graph(graph: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f)


