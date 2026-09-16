import pytest

from mayakb.storage.joblog import InMemoryJobLog, PostgresJobLog


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
def joblog(request):
    if request.param == "memory":
        return InMemoryJobLog()
    return PostgresJobLog(request.getfixturevalue("pg_pool"))


def start(joblog, job_id="j1", kb="kb1", stage="chunk_document"):
    joblog.start(job_id=job_id, kb=kb, stage=stage, input_ids=["d1"], implementation="fixed@1", params_hash="p")


def test_start_marks_running(joblog):
    start(joblog)
    record = joblog.get("j1")
    assert record.status == "running"
    assert record.attempts == 1
    assert record.input_ids == ["d1"]
    assert record.started_at is not None


def test_finish_and_restart_keeps_output_and_counts_attempts(joblog):
    start(joblog)
    joblog.finish("j1", status="succeeded", output_id="out1")
    done = joblog.get("j1")
    assert (done.status, done.output_id) == ("succeeded", "out1")
    assert done.finished_at is not None
    start(joblog)
    again = joblog.get("j1")
    assert again.attempts == 2
    assert again.output_id == "out1"
    assert again.status == "running"


def test_fail_records_error(joblog):
    start(joblog)
    joblog.fail("j1", "ValueError: bad pdf")
    record = joblog.get("j1")
    assert record.status == "failed"
    assert record.error == "ValueError: bad pdf"


def test_list_filters_by_kb_and_status(joblog):
    start(joblog, "j1")
    start(joblog, "j2")
    start(joblog, "j3", kb="other")
    joblog.fail("j2", "boom")
    assert {r.job_id for r in joblog.list("kb1")} == {"j1", "j2"}
    assert [r.job_id for r in joblog.list("kb1", status="failed")] == ["j2"]
    assert len(joblog.list("kb1", limit=1)) == 1


def test_unknown_job(joblog):
    assert joblog.get("nope") is None
