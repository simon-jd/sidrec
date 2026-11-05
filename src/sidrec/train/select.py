"""Pick the tuning trial with the best validation NDCG@10 (docs/protocol.md, section 3).

python -m sidrec.train.select runs/sasrec      # prints the trial number
"""

from __future__ import annotations

import json
import sys

from sidrec.train.common import RESULTS


def best_trial(run_dir: str, expected: int = 12) -> int:
    runs = []
    for path in sorted((RESULTS / run_dir).glob("trial[0-9][0-9].json")):
        record = json.loads(path.read_text())
        if record["phase"] == "tune":
            runs.append((record["valid"]["ndcg@10"], -record["trial"], record["trial"]))
    if len(runs) != expected:
        raise SystemExit(f"{run_dir}: {len(runs)} tuning trials found, expected {expected}")
    return max(runs)[2]


if __name__ == "__main__":
    print(best_trial(sys.argv[1]))
