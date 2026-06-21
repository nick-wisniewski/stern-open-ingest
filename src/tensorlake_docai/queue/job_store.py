# SPDX-License-Identifier: Apache-2.0
"""Tiny file-backed job store for pre-production queue runs."""

from __future__ import annotations

import json
from pathlib import Path

from tensorlake_docai.queue.models import JobRecord


class FileJobStore:
    def __init__(self, root: str | Path = "debug/jobs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def create(self, record: JobRecord) -> JobRecord:
        self.save(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        path = self.path_for(job_id)
        return JobRecord.model_validate_json(path.read_text())

    def save(self, record: JobRecord) -> None:
        self.path_for(record.job_id).write_text(record.model_dump_json(indent=2))

    def save_result(self, job_id: str, result: dict) -> JobRecord:
        record = self.get(job_id)
        record.result = result
        record.error = None
        record.mark(record.status)
        self.save(record)
        return record

    def save_error(self, job_id: str, error: str) -> JobRecord:
        record = self.get(job_id)
        record.error = error
        record.mark(record.status)
        self.save(record)
        return record

    def raw(self, job_id: str) -> dict:
        return json.loads(self.path_for(job_id).read_text())
