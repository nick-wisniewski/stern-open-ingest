# SPDX-License-Identifier: Apache-2.0
"""Tests for pre-production queue job state."""

from tensorlake_docai.queue.job_store import FileJobStore
from tensorlake_docai.queue.models import JobRecord, JobStatus, QueueName
from tensorlake_docai.queue.receiver import _redact_job_response
from tensorlake_docai.queue.redis_streams import RedisStreamsQueue, payload_from_raw


def test_file_job_store_round_trips_job_record(tmp_path):
    store = FileJobStore(tmp_path)
    record = JobRecord(
        request={
            "file_name": "x.pdf",
            "mime_type": "application/pdf",
            "file_bytes": "aGVsbG8=",
        }
    )

    store.create(record)
    loaded = store.get(record.job_id)

    assert loaded.job_id == record.job_id
    assert loaded.status == JobStatus.QUEUED
    assert loaded.queue == QueueName.CPU_INGEST


def test_payload_from_raw_sets_job_id_and_attempts():
    payload = payload_from_raw({"file_name": "x.pdf"}, "job-1", attempts=2)

    assert payload.job_id == "job-1"
    assert payload.attempts == 2
    assert payload.request["file_name"] == "x.pdf"


class _TimeoutClient:
    def xgroup_create(self, *args, **kwargs):
        return None

    def xreadgroup(self, *args, **kwargs):
        raise TimeoutError("timed out")


def test_redis_dequeue_timeout_returns_none():
    queue = RedisStreamsQueue.__new__(RedisStreamsQueue)
    queue.client = _TimeoutClient()
    queue.group = "test"
    queue.max_attempts = 3
    queue._timeout_error = TimeoutError

    assert queue.dequeue([QueueName.CPU_INGEST], consumer="test") is None


def test_redact_job_response_hides_file_bytes():
    response = _redact_job_response(
        {
            "job_id": "job-1",
            "request": {
                "file_name": "x.pdf",
                "file_bytes": "aGVsbG8=",
            },
        }
    )

    assert response["request"]["file_bytes"] == "<redacted base64 bytes: 8 chars>"
