import boto3
import pytest
from moto import mock_aws

from mayakb.core import registry
from mayakb.core.ids import content_id
from mayakb.ingest.connectors.s3 import S3Connector, parse_s3_uri


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="docs")
        client.put_object(Bucket="docs", Key="reports/q2.pdf", Body=b"%PDF-1.7 q2")
        client.put_object(Bucket="docs", Key="reports/notes #1.md", Body=b"# Notes")
        client.put_object(Bucket="docs", Key="reports/image.png", Body=b"png")
        client.put_object(Bucket="docs", Key="reports/folder/", Body=b"")
        client.put_object(Bucket="docs", Key="other/x.txt", Body=b"x")
        yield client


def test_lists_supported_objects_under_prefix(s3):
    refs = list(S3Connector(bucket="docs", prefix="reports/", client=s3).list())
    assert sorted(ref.uri for ref in refs) == ["s3://docs/reports/notes #1.md", "s3://docs/reports/q2.pdf"]
    assert all(ref.etag and '"' not in ref.etag for ref in refs)
    assert all(ref.modified_at is not None and ref.size for ref in refs)


def test_fetch(s3):
    connector = S3Connector(bucket="docs", prefix="reports/", client=s3)
    ref = next(r for r in connector.list() if r.uri.endswith(".md"))
    raw, data = connector.fetch(ref)
    assert data == b"# Notes"
    assert raw.id == content_id(b"# Notes")
    assert raw.mime == "text/markdown"
    assert raw.metadata["filename"] == "notes #1.md"


def test_parse_s3_uri():
    assert parse_s3_uri("s3://docs/a/b #1.pdf") == ("docs", "a/b #1.pdf")
    for bad in ["https://x/y", "s3://bucket-only", "s3:///key"]:
        with pytest.raises(ValueError):
            parse_s3_uri(bad)


def test_registered_as_s3():
    assert registry.get("connector", "s3") is S3Connector
