import json
from typing import Dict, List


def load_decls(path: str) -> List[Dict]:
    decls: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            decls.append(json.loads(line))
    return decls


def build_decl_map(decls: List[Dict]) -> Dict[str, Dict]:
    return {d["decl_name"]: d for d in decls}

