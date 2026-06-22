# SPDX-License-Identifier: Apache-2.0
"""Storage adapters for provider state and artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Protocol

from tensorlake_docai.provider_state.models import ProviderJobState


class ObjectStorage(Protocol):
    def get_bytes(self, key: str) -> bytes: ...

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None: ...

    def get_json(self, key: str) -> dict[str, Any]: ...

    def put_json(self, key: str, data: dict[str, Any]) -> None: ...


class InMemoryProviderStorage:
    def __init__(self):
        self.objects: dict[str, tuple[bytes, str | None]] = {}

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key][0]

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        self.objects[key] = (data, content_type)

    def get_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(key).decode())

    def put_json(self, key: str, data: dict[str, Any]) -> None:
        self.put_bytes(
            key,
            json.dumps(data, indent=2, sort_keys=True).encode(),
            content_type="application/json",
        )


class S3ProviderStorage:
    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name or os.getenv("AWS_REGION"),
        )

    @classmethod
    def from_env(cls) -> "S3ProviderStorage":
        bucket = os.environ["PROVIDER_STATE_BUCKET"]
        return cls(
            bucket,
            prefix=os.getenv("PROVIDER_STATE_PREFIX", ""),
            endpoint_url=os.getenv("PROVIDER_STATE_S3_ENDPOINT_URL"),
            region_name=os.getenv("AWS_REGION"),
        )

    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return response["Body"].read()

    def put_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": self._key(key), "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)

    def get_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(key).decode())

    def put_json(self, key: str, data: dict[str, Any]) -> None:
        self.put_bytes(
            key,
            json.dumps(data, indent=2, sort_keys=True).encode(),
            content_type="application/json",
        )


class ProviderStateStore:
    def __init__(self, storage: ObjectStorage):
        self.storage = storage

    @staticmethod
    def state_key(provider_job_id: str) -> str:
        return f"jobs/{provider_job_id}/state.json"

    @staticmethod
    def input_key(provider_job_id: str, extension: str) -> str:
        clean_ext = extension if extension.startswith(".") else f".{extension}"
        return f"jobs/{provider_job_id}/input/original{clean_ext}"

    @staticmethod
    def result_key(provider_job_id: str) -> str:
        return f"jobs/{provider_job_id}/result/document.json"

    @staticmethod
    def markdown_key(provider_job_id: str) -> str:
        return f"jobs/{provider_job_id}/result/document.md"

    def create_state(self, state: ProviderJobState) -> ProviderJobState:
        self.save_state(state)
        return state

    def get_state(self, provider_job_id: str) -> ProviderJobState:
        return ProviderJobState.model_validate(
            self.storage.get_json(self.state_key(provider_job_id))
        )

    def save_state(self, state: ProviderJobState) -> None:
        self.storage.put_json(
            self.state_key(state.provider_job_id),
            state.model_dump(mode="json"),
        )

    def update_state(
        self,
        provider_job_id: str,
        updater: Callable[[ProviderJobState], ProviderJobState],
    ) -> ProviderJobState:
        current = self.get_state(provider_job_id)
        updated = updater(current)
        self.save_state(updated)
        return updated

    def put_input(
        self,
        provider_job_id: str,
        extension: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> str:
        key = self.input_key(provider_job_id, extension)
        self.storage.put_bytes(key, data, content_type=content_type)
        return key

    def get_input(self, state: ProviderJobState) -> bytes:
        if not state.input_object_key:
            raise ValueError(f"Provider job {state.provider_job_id} has no input object key")
        return self.storage.get_bytes(state.input_object_key)

    def put_result(self, provider_job_id: str, result: dict[str, Any]) -> tuple[str, str]:
        result_key = self.result_key(provider_job_id)
        markdown_key = self.markdown_key(provider_job_id)
        markdown = ""
        document = result.get("document")
        if isinstance(document, dict):
            markdown = document.get("document_markdown") or ""

        self.storage.put_json(result_key, result)
        self.storage.put_bytes(markdown_key, markdown.encode(), content_type="text/markdown")
        return result_key, markdown_key
