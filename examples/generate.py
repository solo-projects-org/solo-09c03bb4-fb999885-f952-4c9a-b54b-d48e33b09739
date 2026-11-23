"""Generate demo CSVs for the split-verification service.

Usage::

    python examples/generate.py [output_dir]

The example round has three periods (300 / 300 / 400 users).  Period 2 is
deliberately skewed towards group B so the verification page shows a flag.
"""

from __future__ import annotations

import csv
import os
import random
import sys

PLAN = [
    # period, users, planned B probability, actual B probability used in demo
    (1, 300, 0.5, 0.50),
    (2, 300, 0.4, 0.85),
    (3, 400, 0.6, 0.60),
]


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(20261122)
    users = [f"u{i:04d}" for i in range(sum(row[1] for row in PLAN))]
    rng.shuffle(users)

    cursor = 0
    for period, count, _planned, actual in PLAN:
        path = os.path.join(out_dir, f"period-{period}.csv")
        group_b = 0
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["user_id", "group"])
            for _ in range(count):
                group = "B" if rng.random() < actual else "A"
                group_b += group == "B"
                writer.writerow([users[cursor], group])
                cursor += 1
        print(f"{path}: {count} users, {group_b} in B")


if __name__ == "__main__":
    main()
