# SPDX-License-Identifier: Apache-2.0
"""Small Redis Streams queue wrapper with leases, retries, and dead letters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tensorlake_docai.queue.models import QueueName, QueuePayload


@dataclass(frozen=True)
class QueueMessage:
    queue: QueueName
    message_id: str
    payload: QueuePayload


class RedisStreamsQueue:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        group: str = "stern-open-ingest",
        max_attempts: int = 3,
    ):
        try:
            import redis
        except ImportError as e:
            raise RuntimeError("Install the `redis` package to use Redis queue workers.") from e

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.group = group
        self.max_attempts = max_attempts

    def enqueue(self, queue: QueueName, payload: QueuePayload) -> str:
        self._ensure_group(queue)
        return self.client.xadd(queue.value, {"payload": payload.model_dump_json()})

    def dequeue(
        self,
        queues: list[QueueName],
        *,
        consumer: str,
        block_ms: int = 5000,
        count: int = 1,
    ) -> QueueMessage | None:
        for queue in queues:
            self._ensure_group(queue)

        streams = {queue.value: ">" for queue in queues}
        response = self.client.xreadgroup(
            self.group,
            consumer,
            streams,
            count=count,
            block=block_ms,
        )
        if not response:
            return None

        stream_name, messages = response[0]
        message_id, fields = messages[0]
        return QueueMessage(
            queue=QueueName(stream_name),
            message_id=message_id,
            payload=QueuePayload.model_validate_json(fields["payload"]),
        )

    def ack(self, message: QueueMessage) -> None:
        self.client.xack(message.queue.value, self.group, message.message_id)

    def retry_or_dead_letter(self, message: QueueMessage, *, error: str) -> str:
        next_payload = message.payload.model_copy(update={"attempts": message.payload.attempts + 1})
        if next_payload.attempts >= self.max_attempts:
            target = f"{message.queue.value}-dead"
            self.client.xadd(
                target,
                {
                    "payload": next_payload.model_dump_json(),
                    "error": error,
                    "source_queue": message.queue.value,
                },
            )
        else:
            target = message.queue.value
            self.enqueue(message.queue, next_payload)

        self.ack(message)
        return target

    def _ensure_group(self, queue: QueueName) -> None:
        try:
            self.client.xgroup_create(queue.value, self.group, id="0", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise


def payload_from_raw(raw: dict[str, Any], job_id: str, attempts: int = 0) -> QueuePayload:
    return QueuePayload(job_id=job_id, request=raw, attempts=attempts)
