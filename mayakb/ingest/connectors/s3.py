"""Objects in an S3 bucket. Credentials come from the standard AWS chain (IAM role on EC2)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from mayakb.core.contracts import RawDocument, SourceRef
from mayakb.core.ids import content_id
from mayakb.ingest.connectors.base import DEFAULT_SUFFIXES, Connector, guess_mime


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3://bucket/key uri: {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ValueError(f"not an s3://bucket/key uri: {uri!r}")
    return bucket, key


class S3Connector(Connector):
    name = "s3"
    version = "1"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        suffixes: list[str] | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        client: Any = None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.suffixes = tuple(s.lower() for s in (suffixes or DEFAULT_SUFFIXES))
        if client is None:
            import boto3

            client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
        self.client = client

    def list(self) -> Iterator[SourceRef]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or not key.lower().endswith(self.suffixes):
                    continue
                yield SourceRef(
                    uri=f"s3://{self.bucket}/{key}",
                    etag=obj["ETag"].strip('"'),
                    modified_at=obj["LastModified"],
                    size=obj["Size"],
                )

    def fetch(self, ref: SourceRef) -> tuple[RawDocument, bytes]:
        bucket, key = parse_s3_uri(ref.uri)
        data = self.client.get_object(Bucket=bucket, Key=key)["Body"].read()
        raw = RawDocument(
            id=content_id(data),
            source_uri=ref.uri,
            connector=self.name,
            mime=guess_mime(key),
            etag=ref.etag,
            modified_at=ref.modified_at,
            metadata={"size": len(data), "filename": key.rsplit("/", 1)[-1]},
        )
        return raw, data
