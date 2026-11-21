"""Concurrency guarantees around masking and publishing.

The expensive suppression analysis must run without holding SQLite's
database-wide write lock.  These tests pause ``suppression.analyze`` and check
that other work still makes progress, that a stale snapshot is rejected without
writing anything, and that simultaneous mask saves on one detail version are
all accepted with strictly increasing mask versions.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app import suppression

Q1 = "2024Q1"
Q2 = "2024Q2"

DISTRICTS = ["东城区", "西城区", "南城区", "北城区"]
INDUSTRIES = ["制造业", "建筑业", "批发零售"]

VALUES = [
    [30, 50, 40],
    [25, 35, 35],
    [10, 20, 30],
    [45, 3, 2],
]
ROW_TOTALS = [sum(r) for r in VALUES]
COL_TOTALS = [sum(VALUES[i][j] for i in range(len(VALUES)))
              for j in range(len(VALUES[0]))]

# Same margins, different cell values: a legitimate new detail version.
ALT_VALUES = [
    [29, 51, 40],
    [26, 34, 35],
    [10, 20, 30],
    [45, 3, 2],
]


def register(client, quarter):
    body = {
        "districts": [{"name": n, "total": t}
                      for n, t in zip(DISTRICTS, ROW_TOTALS)],
        "industries": [{"name": n, "total": t}
                       for n, t in zip(INDUSTRIES, COL_TOTALS)],
    }
    response = client.put(
        f"/api/v1/editor/quarters/{quarter}/totals", json=body)
    assert response.status_code == 200, response.text


def submit(client, quarter, values=VALUES, **extra):
    body = {"values": values, "must_public": []}
    body.update(extra)
    response = client.post(
        f"/api/v1/editor/quarters/{quarter}/detail", json=body)
    assert response.status_code == 200, response.text
    return response


def save_mask(client, quarter, cells, **extra):
    body = {"cells": [list(c) for c in cells]}
    body.update(extra)
    return client.put(
        f"/api/v1/editor/quarters/{quarter}/mask", json=body)


def publish(client, quarter, **extra):
    return client.post(
        f"/api/v1/editor/quarters/{quarter}/publish", json=extra)


def valid_mask_cells(client, quarter):
    response = client.post(
        f"/api/v1/editor/quarters/{quarter}/suppression-advice", json={})
    assert response.status_code == 200, response.text
    advice = response.json()
    assert advice["valid"], advice
    return [(c["row"], c["col"]) for c in advice["masked"]]


class Gate:
    """Let the first ``analyze`` call block until the test releases it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._first = True

    def install(self, monkeypatch) -> None:
        original = suppression.analyze

        def analyze(*args, **kwargs):
            with self._lock:
                is_first = self._first
                self._first = False
            if is_first:
                self.started.set()
                if not self.release.wait(15):
                    raise AssertionError("test did not release the analysis")
            return original(*args, **kwargs)

        monkeypatch.setattr(suppression, "analyze", analyze)

    def wait_started(self) -> None:
        assert self.started.wait(15), "analysis never started"


def in_thread(worker):
    result: dict = {}

    def run():
        try:
            result["value"] = worker()
        except BaseException as exc:  # pragma: no cover - surfaced below
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    return thread, result


def finish(thread, result):
    thread.join(20)
    assert not thread.is_alive(), "request thread did not finish"
    if "error" in result:
        raise result["error"]
    return result["value"]


# --------------------------------------------------------------- no blocking

def test_mask_analysis_does_not_block_other_quarters(client, monkeypatch):
    for quarter in (Q1, Q2):
        register(client, quarter)
        submit(client, quarter, VALUES)
    cells = valid_mask_cells(client, Q1)

    gate = Gate()
    gate.install(monkeypatch)
    thread, result = in_thread(lambda: save_mask(client, Q1, cells))

    gate.wait_started()
    # Still blocked in analysis: another quarter must accept a detail.
    submit(client, Q2, ALT_VALUES)
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 200, response.text


def test_mask_analysis_does_not_block_another_mask(client, monkeypatch):
    register(client, Q1)
    submit(client, Q1, VALUES)
    cells = valid_mask_cells(client, Q1)

    gate = Gate()
    gate.install(monkeypatch)
    thread, result = in_thread(lambda: save_mask(client, Q1, cells))

    gate.wait_started()
    # Still blocked in analysis: the same quarter may save another mask.
    concurrent = save_mask(client, Q1, cells)
    assert concurrent.status_code == 200, concurrent.text
    first_version = concurrent.json()["mask_version"]
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 200, response.text
    assert response.json()["mask_version"] == first_version + 1


def test_mask_analysis_does_not_block_another_publish(client, monkeypatch):
    cells_by_quarter = {}
    for quarter in (Q1, Q2):
        register(client, quarter)
        submit(client, quarter, VALUES)
        cells_by_quarter[quarter] = valid_mask_cells(client, quarter)
    assert save_mask(client, Q2, cells_by_quarter[Q2]).status_code == 200

    gate = Gate()
    gate.install(monkeypatch)
    thread, result = in_thread(
        lambda: save_mask(client, Q1, cells_by_quarter[Q1]))

    gate.wait_started()
    # Still blocked in analysis: another quarter can complete its publish.
    other = publish(client, Q2)
    assert other.status_code == 200, other.text
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 200, response.text
    assert client.get(f"/api/v1/bulletins/{Q1}").status_code == 404


def test_publish_analysis_does_not_block_other_quarters(client, monkeypatch):
    for quarter in (Q1, Q2):
        register(client, quarter)
        submit(client, quarter, VALUES)
        cells = valid_mask_cells(client, quarter)
        assert save_mask(client, quarter, cells).status_code == 200

    gate = Gate()
    gate.install(monkeypatch)
    thread, result = in_thread(lambda: publish(client, Q1))

    gate.wait_started()
    # Still blocked in analysis: another quarter can be published.
    other = publish(client, Q2)
    assert other.status_code == 200, other.text
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 200, response.text


# --------------------------------------------------- stale snapshot conflicts

def test_mask_for_superseded_detail_is_rejected_without_writing(
        client, monkeypatch):
    register(client, Q1)
    submit(client, Q1, VALUES)
    cells = valid_mask_cells(client, Q1)

    gate = Gate()
    gate.install(monkeypatch)
    # No detail_version in the request: the stale snapshot must still conflict.
    thread, result = in_thread(lambda: save_mask(client, Q1, cells))

    gate.wait_started()
    submit(client, Q1, ALT_VALUES)          # detail v2 while analysis runs
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stale_detail_version"

    state = client.get(f"/api/v1/editor/quarters/{Q1}").json()
    assert state["current_detail_version"] == 2
    assert state["mask_version"] is None
    assert state["masked_cells"] == []


def test_publish_for_superseded_detail_is_rejected_without_writing(
        client, monkeypatch):
    register(client, Q1)
    submit(client, Q1, VALUES)
    assert save_mask(client, Q1, valid_mask_cells(client, Q1)).status_code == 200

    gate = Gate()
    gate.install(monkeypatch)
    # No versions supplied: the review snapshot is what matters.
    thread, result = in_thread(lambda: publish(client, Q1))

    gate.wait_started()
    submit(client, Q1, ALT_VALUES)          # detail v2 while review runs
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stale_detail_version"
    assert client.get(f"/api/v1/bulletins/{Q1}").status_code == 404


def test_publish_for_superseded_mask_is_rejected_without_writing(
        client, monkeypatch):
    register(client, Q1)
    submit(client, Q1, VALUES)
    cells = valid_mask_cells(client, Q1)
    assert save_mask(client, Q1, cells).status_code == 200

    gate = Gate()
    gate.install(monkeypatch)
    thread, result = in_thread(lambda: publish(client, Q1))

    gate.wait_started()
    revised = save_mask(client, Q1, cells)  # mask version changes mid-review
    assert revised.status_code == 200, revised.text
    gate.release.set()

    response = finish(thread, result)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stale_mask_version"
    assert client.get(f"/api/v1/bulletins/{Q1}").status_code == 404

    # Nothing was written, so a fresh review can still publish.
    assert publish(client, Q1).status_code == 200


# ------------------------------------------------------ concurrent mask saves

@pytest.mark.parametrize("workers", [2, 5])
def test_concurrent_mask_saves_all_succeed_in_write_order(
        client, monkeypatch, workers):
    register(client, Q1)
    submit(client, Q1, VALUES)
    cells = valid_mask_cells(client, Q1)

    original = suppression.analyze
    barrier = threading.Barrier(workers)

    def synchronized_analyze(*args, **kwargs):
        # Every request must reach the analysis without holding a write lock;
        # any serialisation here would time the barrier out.
        barrier.wait(timeout=20)
        return original(*args, **kwargs)

    monkeypatch.setattr(suppression, "analyze", synchronized_analyze)

    token = client.editor_token
    threads = []
    results = []
    for _ in range(workers):
        holder: dict = {}

        def worker(holder=holder):
            with TestClient(client.app) as own:
                own.headers.update({"Authorization": f"Bearer {token}"})
                holder["response"] = save_mask(own, Q1, cells)

        thread = threading.Thread(target=worker)
        thread.start()
        threads.append(thread)
        results.append(holder)

    for thread in threads:
        thread.join(30)
        assert not thread.is_alive(), "a concurrent mask save did not finish"

    monkeypatch.setattr(suppression, "analyze", original)

    versions = set()
    for holder in results:
        response = holder["response"]
        assert response.status_code == 200, response.text
        versions.add(response.json()["mask_version"])
    assert versions == set(range(1, workers + 1))

    state = client.get(f"/api/v1/editor/quarters/{Q1}").json()
    assert state["mask_version"] == workers

    # The quarter still publishes exactly once with the winning mask.
    assert publish(client, Q1).status_code == 200
    assert publish(client, Q1).json()["error"]["code"] == "already_published"
