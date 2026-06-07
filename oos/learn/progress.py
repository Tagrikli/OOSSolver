"""Run reporting — `progress.md` (human-readable) + `metrics.jsonl` (machine).

The *shape* of what trainers report doesn't change run to run: a header of
static run facts, then a growing table of per-eval (or per-iter) records. So
this is one reusable writer instead of a bespoke `_write_progress` per trainer.

    writer = ProgressWriter(
        run_dir, title="retrieve — myrun",
        meta=["facility tiny", "reward DELIVER 1.0 (only)"],
        columns=[
            Column("iter", "iter"),
            Column("env_steps", "env_steps", lambda v: f"{v:,}"),
            Column("greedy", "greedy solve", pct),
            Column("sampled", "sampled", pct),
        ],
    )
    writer.record(iter=30, env_steps=30720, greedy=0.82, sampled=0.4)

Each `record(...)` appends the raw dict to `metrics.jsonl` and rewrites
`progress.md` (header + the full table so far). `metrics.jsonl` keeps the raw
numbers; the table uses each column's `fmt` for display only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def pct(v: float) -> str:
    """Format a 0..1 rate as a percentage (handles nan)."""
    return f"{v * 100:.1f}%"


@dataclass
class Column:
    key: str
    header: str
    fmt: Callable[[Any], str] = str


@dataclass
class ProgressWriter:
    run_dir: "str | Path"
    title: str
    columns: list[Column]
    meta: list[str] = field(default_factory=list)
    # Resume handling. `resume_at=None` → fresh run: start metrics.jsonl clean
    # (don't inherit a prior run's rows). `resume_at=<iter>` → keep prior rows
    # with `key_field < resume_at` (drops any checkpoint-rollback overlap and
    # de-dups), so progress.md continues the full curve and metrics.jsonl stays
    # monotonic. `key_field` is the per-row field used to order/dedup.
    resume_at: "int | None" = None
    key_field: str = "iter"
    _history: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        mpath = run_dir / "metrics.jsonl"
        if self.resume_at is None:
            mpath.write_text("")                       # fresh: clean slate
        elif mpath.exists():
            kept: dict[int, dict] = {}                 # resume: keep < resume_at
            for line in mpath.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                k = int(row.get(self.key_field, -1))
                if k < self.resume_at:
                    kept[k] = row                      # last-wins → de-dups
            self._history = [kept[k] for k in sorted(kept)]
            mpath.write_text("".join(json.dumps(r) + "\n" for r in self._history))
        (run_dir / "progress.md").write_text(self._render() + "\n")

    def record(self, **row: Any) -> None:
        """Append one record: raw dict → metrics.jsonl, and rewrite progress.md."""
        self._history.append(row)
        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        (run_dir / "progress.md").write_text(self._render() + "\n")

    def _render(self) -> str:
        lines = [f"# {self.title}", ""]
        lines += [f"- {m}" for m in self.meta]
        lines += [
            "",
            "| " + " | ".join(c.header for c in self.columns) + " |",
            "|" + "|".join("---" for _ in self.columns) + "|",
        ]
        for row in self._history:
            cells = [c.fmt(row.get(c.key)) if c.key in row else "" for c in self.columns]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
