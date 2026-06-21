# SPDX-License-Identifier: Apache-2.0
"""Queue payload and job-state models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class QueueName(StrEnum):
    CPU_INGEST = "cpu-ingest"
    GPU_OCR = "gpu-ocr"
    CPU_POSTPROCESS = "cpu-postprocess"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class QueuePayload(BaseModel):
    job_id: str
    request: dict[str, Any]
    attempts: int = 0


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    queue: QueueName = QueueName.CPU_INGEST
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def mark(self, status: JobStatus, *, queue: QueueName | None = None) -> "JobRecord":
        self.status = status
        if queue is not None:
            self.queue = queue
        self.updated_at = datetime.now(timezone.utc)
        return self
