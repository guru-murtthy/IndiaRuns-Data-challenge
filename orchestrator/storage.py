from __future__ import annotations

from collections import OrderedDict

from orchestrator.models import RunResult


class RunStore:
    def __init__(self) -> None:
        self._runs: "OrderedDict[str, RunResult]" = OrderedDict()

    def save(self, run: RunResult) -> None:
        self._runs[run.run_id] = run
        self._runs.move_to_end(run.run_id, last=False)
        while len(self._runs) > 8:
            self._runs.popitem(last=True)

    def get(self, run_id: str) -> RunResult | None:
        return self._runs.get(run_id)

    def list_recent(self) -> list[RunResult]:
        return list(self._runs.values())


run_store = RunStore()
