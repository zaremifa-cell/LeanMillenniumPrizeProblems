import re
from typing import Iterable, List

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|\d+|\S")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def avg_len(tokens: Iterable[str]) -> float:
    toks = list(tokens)
    return float(sum(len(t) for t in toks)) / max(len(toks), 1)

