# SPDX-License-Identifier: Apache-2.0
"""Provider job state models for S3-backed orchestration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl


class ProviderJobStatus(StrEnum):
    QUEUED = "queued"
    INPUT_STORED = "input_stored"
    CLASSIFYING = "classifying"
    CPU_EXTRACTING = "cpu_extracting"
    GPU_OCR_QUEUED = "gpu_ocr_queued"
    GPU_OCR_RUNNING = "gpu_ocr_running"
    POSTPROCESSING = "postprocessing"
    RESULT_STORED = "result_stored"
    WEBHOOK_PENDING = "webhook_pending"
    WEBHOOK_DELIVERED = "webhook_delivered"
    WEBHOOK_RETRYING = "webhook_retrying"
    FAILED = "failed"


TERMINAL_STATUSES = {
    ProviderJobStatus.WEBHOOK_DELIVERED,
    ProviderJobStatus.FAILED,
}

ALLOWED_TRANSITIONS: dict[ProviderJobStatus, set[ProviderJobStatus]] = {
    ProviderJobStatus.QUEUED: {
        ProviderJobStatus.INPUT_STORED,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.INPUT_STORED: {
        ProviderJobStatus.CLASSIFYING,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.CLASSIFYING: {
        ProviderJobStatus.CPU_EXTRACTING,
        ProviderJobStatus.GPU_OCR_QUEUED,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.CPU_EXTRACTING: {
        ProviderJobStatus.POSTPROCESSING,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.GPU_OCR_QUEUED: {
        ProviderJobStatus.GPU_OCR_RUNNING,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.GPU_OCR_RUNNING: {
        ProviderJobStatus.POSTPROCESSING,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.POSTPROCESSING: {
        ProviderJobStatus.RESULT_STORED,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.RESULT_STORED: {
        ProviderJobStatus.WEBHOOK_PENDING,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.WEBHOOK_PENDING: {
        ProviderJobStatus.WEBHOOK_DELIVERED,
        ProviderJobStatus.WEBHOOK_RETRYING,
        ProviderJobStatus.FAILED,
    },
    ProviderJobStatus.WEBHOOK_RETRYING: {
        ProviderJobStatus.WEBHOOK_DELIVERED,
        ProviderJobStatus.WEBHOOK_RETRYING,
        ProviderJobStatus.FAILED,
    },
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def default_expires_at() -> datetime:
    return utcnow() + timedelta(days=7)


class ProviderParseRequest(BaseModel):
    file_url: HttpUrl
    webhook_url: HttpUrl
    ocr_model: str = "paddle-ocr-vl"


class ProviderPageClassification(BaseModel):
    page_number: int
    route: str
    reason: str
    text_chars: int = 0
    image_area_ratio: float = 0.0


class ProviderJobState(BaseModel):
    provider_job_id: str = Field(default_factory=lambda: uuid4().hex)
    external_job_id: str | None = None
    status: ProviderJobStatus = ProviderJobStatus.QUEUED
    input_object_key: str | None = None
    result_object_key: str | None = None
    markdown_object_key: str | None = None
    ocr_pages: list[int] | None = None
    total_pages: int | None = None
    page_classification: list[ProviderPageClassification] = Field(default_factory=list)
    parse_request: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str | None = None
    webhook_attempts: int = 0
    last_error: str | None = None
    version: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime = Field(default_factory=default_expires_at)

    def transition(
        self,
        status: ProviderJobStatus,
        *,
        now: datetime | None = None,
        **updates: Any,
    ) -> "ProviderJobState":
        if status != self.status and status not in ALLOWED_TRANSITIONS.get(self.status, set()):
            raise ValueError(f"Invalid provider state transition: {self.status} -> {status}")

        next_state = self.model_copy(update=updates)
        next_state.status = status
        next_state.version = self.version + 1
        next_state.updated_at = now or utcnow()
        return next_state

    def redacted_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if data.get("webhook_url"):
            data["webhook_url"] = "<redacted>"
        return data
