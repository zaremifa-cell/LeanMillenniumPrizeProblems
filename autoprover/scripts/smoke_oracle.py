import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoprover.utils.lean_oracle import LeanOracle


def main() -> None:
    oracle = LeanOracle()
    try:
        # Simple exact proof: True.intro : True
        resp = oracle.request("True", [], {"kind": "exact", "lemma": "True.intro"})
        print("exact True.intro:", resp)

        # Simple intro: goal becomes n = n
        resp = oracle.request("∀ n : Nat, n = n", [], {"kind": "intro"})
        print("intro:", resp)
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
