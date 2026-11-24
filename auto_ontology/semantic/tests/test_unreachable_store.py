# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A build that can't reach Postgres fails instead of finishing damaged (#553).

At the server's connection limit, every handler that skips one bad table or
column also skipped the "too many clients" error, and the build finished with
missing semantic FKs and stale embeddings. Errors about one table are still
skipped.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import psycopg
import pytest
import sqlalchemy.exc

from auto_ontology.semantic import deterministic, pipeline
from auto_ontology.semantic.cancellation import clear_cancel
from auto_ontology.semantic.models import ProcessTableResult


def _too_many_clients() -> sqlalchemy.exc.OperationalError:
    return sqlalchemy.exc.OperationalError(
        "SELECT 1",
        {},
        psycopg.OperationalError(
            "connection failed: FATAL:  sorry, too many clients already"
        ),
    )


@pytest.fixture(autouse=True)
def _clean_flag() -> Iterator[None]:
    clear_cancel()
    yield
    clear_cancel()


def _tables(n: int) -> list[dict[str, Any]]:
    return [{"id": f"t{i}", "name": f"table_{i}"} for i in range(n)]


def _compile(process_table: Any, n: int) -> int:
    with (
        patch.object(
            pipeline, "fetch_all_tables_without_term", return_value=_tables(n)
        ),
        patch.object(
            pipeline, "fetch_table_context", return_value={"columns": [{"n": 1}]}
        ),
        patch.object(pipeline, "load_domain_summary", return_value=None),
        patch.object(pipeline, "is_distinct_value_probing_enabled", return_value=False),
        patch.object(pipeline, "reset_sampling_breaker"),
        patch.object(pipeline, "process_table", process_table),
    ):
        return pipeline.compile_semantic_layer("db")


def test_a_table_that_cant_reach_postgres_fails_the_compilation() -> None:
    processed: list[str] = []
    lock = threading.Lock()

    def _fake_process_table(
        table: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
    ) -> None:
        with lock:
            processed.append(table["name"])
        if table["name"] == "table_0":
            raise _too_many_clients()
        return None

    with pytest.raises(sqlalchemy.exc.OperationalError):
        _compile(_fake_process_table, 25)

    # The tables already running finish; the queued ones aren't started, since
    # each would spend its LLM calls and then fail the same way.
    assert len(processed) < 25


def test_an_error_about_one_table_still_skips_only_that_table() -> None:
    processed: list[str] = []
    lock = threading.Lock()

    def _fake_process_table(
        table: dict[str, Any], ctx: dict[str, Any], **kwargs: Any
    ) -> ProcessTableResult:
        with lock:
            processed.append(table["name"])
        if table["name"] == "table_0":
            raise ValueError("the LLM returned no terms")
        return ProcessTableResult()

    count = _compile(_fake_process_table, 8)

    assert len(processed) == 8
    assert count == 7


def _describe(refresh_error: BaseException) -> list[str]:
    columns = [{"id": "c1", "name": "amt"}]
    with (
        patch.object(
            deterministic,
            "_generate_column_descriptions",
            return_value={"amt": "Order amount"},
        ),
        patch.object(deterministic, "store_column_descriptions", return_value={"c1"}),
        patch.object(
            deterministic, "refresh_catalog_embeddings", side_effect=refresh_error
        ),
    ):
        return deterministic.fill_missing_column_descriptions("t1", columns)


def test_a_refresh_that_cant_reach_the_vector_store_is_raised() -> None:
    """This handler logged the 49 failed refreshes in the issue."""
    with pytest.raises(sqlalchemy.exc.OperationalError):
        _describe(_too_many_clients())


def test_a_refresh_that_fails_for_another_reason_is_still_logged() -> None:
    assert _describe(RuntimeError("embedding endpoint returned 500")) == ["amt"]
