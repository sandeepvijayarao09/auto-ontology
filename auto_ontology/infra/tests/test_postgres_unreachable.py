# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telling "Postgres can't be reached" apart from "this statement failed" (#553)."""

from __future__ import annotations

import asyncio
import socket

import asyncpg
import psycopg
import psycopg.errors
import pytest
import sqlalchemy.exc

from auto_ontology.infra.postgres import is_postgres_unreachable


def _wrapped(orig: BaseException) -> sqlalchemy.exc.DBAPIError:
    """How SQLAlchemy hands a driver error to the code that called it."""
    return sqlalchemy.exc.OperationalError("SELECT 1", {}, orig)


@pytest.mark.parametrize(
    "exc",
    [
        # What two builds at max_connections logged, through SQLAlchemy.
        _wrapped(
            psycopg.OperationalError(
                "connection failed: FATAL:  sorry, too many clients already"
            )
        ),
        psycopg.errors.TooManyConnections("too many connections for role"),
        psycopg.errors.AdminShutdown("terminating connection"),
        psycopg.errors.CannotConnectNow("the database system is starting up"),
        asyncpg.exceptions.TooManyConnectionsError("sorry, too many clients already"),
        asyncpg.exceptions.ConnectionDoesNotExistError("connection was closed"),
        ConnectionRefusedError(61, "Connect call failed"),
        OSError(
            "Multiple exceptions: [Errno 61] Connect call failed ('::1', 5432), "
            "[Errno 61] Connect call failed ('127.0.0.1', 5432)"
        ),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_connection_failures_are_unreachable(exc: BaseException) -> None:
    assert is_postgres_unreachable(exc)


def test_the_cause_is_found_through_a_wrapping_exception() -> None:
    try:
        try:
            raise asyncpg.exceptions.TooManyConnectionsError("too many clients")
        except asyncpg.exceptions.TooManyConnectionsError as inner:
            raise RuntimeError("write_embedded_rows failed") from inner
    except RuntimeError as outer:
        assert is_postgres_unreachable(outer)


@pytest.mark.parametrize(
    "exc",
    [
        # A statement timeout or a bad statement is about that statement; the
        # per-table handlers keep skipping it.
        _wrapped(psycopg.errors.QueryCanceled("canceling statement due to timeout")),
        psycopg.errors.UniqueViolation("duplicate key"),
        psycopg.errors.UndefinedTable("relation does not exist"),
        psycopg.errors.DiskFull("could not extend file"),
        asyncpg.exceptions.UndefinedColumnError("column does not exist"),
        ValueError("bad LLM output"),
        FileNotFoundError(2, "No such file"),
        OSError("something else"),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_statement_errors_are_not_unreachable(exc: BaseException) -> None:
    assert not is_postgres_unreachable(exc)


@pytest.fixture
def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_a_refused_catalog_connection_is_unreachable(closed_port: int) -> None:
    """The catalog goes through SQLAlchemy and psycopg."""
    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg://u:p@127.0.0.1:{closed_port}/db"
    )
    with pytest.raises(sqlalchemy.exc.OperationalError) as caught:
        engine.connect()
    engine.dispose()
    assert is_postgres_unreachable(caught.value)


def test_a_refused_vector_store_connection_is_unreachable(closed_port: int) -> None:
    """The vector store goes through asyncpg."""

    async def _connect() -> None:
        await asyncpg.connect(f"postgresql://u:p@localhost:{closed_port}/db")

    with pytest.raises(OSError) as caught:
        asyncio.run(_connect())
    assert is_postgres_unreachable(caught.value)
