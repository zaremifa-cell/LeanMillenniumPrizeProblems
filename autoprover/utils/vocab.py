from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


class Vocab:
    def __init__(self, tokens: Iterable[str], max_size: int = 10000) -> None:
        counts = Counter(tokens)
        most_common = [t for t, _ in counts.most_common(max_size - 2)]
        self.token_to_id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for t in most_common:
            if t not in self.token_to_id:
                self.token_to_id[t] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.token_to_id)

