import json
import random
from typing import Dict, List, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def load_graph(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pairs(edges: Dict[str, List[str]], max_pairs: int = 200000) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for src, deps in edges.items():
        for dep in deps:
            pairs.append((src, dep))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def pretrain_embeddings(graph: Dict, out: str, dim: int = 64, epochs: int = 5, negatives: int = 5) -> None:
    nodes = sorted(graph["nodes"].keys())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = {k: v for k, v in graph["edges"].items()}

    pairs = build_pairs(edges)
    pairs_idx = [(node_to_idx[a], node_to_idx[b]) for a, b in pairs if a in node_to_idx and b in node_to_idx]

    if torch is None:
        # fallback: random embeddings
        import numpy as np

        emb = np.random.normal(0, 0.1, size=(len(nodes), dim)).astype("float32")
        np.savez(out, embeddings=emb, node_to_idx=node_to_idx)
        return

    emb = nn.Embedding(len(nodes), dim)
    opt = torch.optim.Adam(emb.parameters(), lr=1e-3)

    for epoch in range(epochs):
        random.shuffle(pairs_idx)
        total_loss = 0.0
        for src, dst in pairs_idx:
            src_t = torch.tensor([src])
            dst_t = torch.tensor([dst])
            pos_score = (emb(src_t) * emb(dst_t)).sum(dim=1)
            pos_loss = torch.nn.functional.logsigmoid(pos_score).mean()

            neg_samples = torch.randint(0, len(nodes), (negatives,))
            neg_score = (emb(src_t) * emb(neg_samples)).sum(dim=1)
            neg_loss = torch.nn.functional.logsigmoid(-neg_score).mean()

            loss = -(pos_loss + neg_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{epochs} loss={total_loss/ max(len(pairs_idx),1):.4f}")

    torch.save({"embeddings": emb.weight.detach(), "node_to_idx": node_to_idx}, out)


