from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from bretez.state import new_id, now_iso


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def start(self, job_type: str, target: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        job_id = new_id("job")
        job = {
            "id": job_id,
            "type": job_type,
            "status": "queued",
            "message": "Queued",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job_id, target), name=f"bretez-{job_type}-{job_id}", daemon=True)
        thread.start()
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No job with id {job_id!r}.")
            return dict(self._jobs[job_id])

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(job) for job in sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)]

    def _run(self, job_id: str, target: Callable[[], dict[str, Any]]) -> None:
        self._update(job_id, status="running", message="Running")
        try:
            result = target()
        except Exception as error:  # pragma: no cover - preserves background failures for API users.
            self._update(job_id, status="failed", message=str(error), error=traceback.format_exc())
            return
        self._update(job_id, status="completed", message="Completed", result=result)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(values)
            job["updated_at"] = now_iso()
