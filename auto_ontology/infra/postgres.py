# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Env-derived Postgres connection config, shared across the codebase.

Not VDB-specific: the server, ingestion service, and dev tools all build the
local Postgres URL from the same ``POSTGRES_*`` env vars via this helper.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import quote

import psycopg

#: Schema holding the tables Prisma owns (`conversations`, `messages`,
#: `acronyms`, `prompts`, `configurations`, `conversation_analytics`, auth).
#:
#: The backend only ever *reads and writes rows* here -- the shape is Prisma's,
#: declared in `frontend/prisma/schema.prisma`, and Alembic deliberately does not
#: manage it. Every backend statement against these tables must qualify them with
#: this constant.
#:
#: Qualifying is not optional politeness. These tables used to live in `public`
#: and were reached unqualified, which worked only because the role's
#: `search_path` happened to resolve there. When they moved to `frontend` every
#: such statement broke -- and mostly broke *quietly*, because the call sites
#: catch their exceptions: the semantic-compilation flag silently read as
#: disabled, and acronyms and custom prompts silently came back empty. A
#: schema-qualified name cannot fail that way.
FRONTEND_SCHEMA = "frontend"


def get_postgres_connection_string() -> str:
    """Build the local Postgres URL from ``POSTGRES_*`` env vars.

    ``database`` overrides ``POSTGRES_DATABASE`` when given (useful for
    multi-DB tools that target several databases on the same instance).
    """
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ.get("POSTGRES_DATABASE", "auto_ontology")
    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{db}"


# SQLSTATEs meaning the server can't be reached or won't take the connection:
# class 08 (connection exception), too_many_connections, and the shutdown /
# not-yet-accepting states. Other errors, such as a statement timeout, are about
# one statement and stay with the caller.
_UNREACHABLE_SQLSTATE_CLASSES = ("08",)
_UNREACHABLE_SQLSTATES = frozenset({"53300", "57P01", "57P02", "57P03"})


def is_postgres_unreachable(exc: BaseException) -> bool:
    """Whether *exc* means Postgres can't be reached, not that one statement failed.

    The semantic build catches errors per table and per column so that one bad
    table doesn't stop the rest. A connection error isn't about that table: at
    the server's connection limit every later table fails the same way, and the
    build used to finish with missing FKs and stale embeddings (#553). Callers
    that swallow errors re-raise these instead.

    The cause is found anywhere in the chain, since SQLAlchemy wraps the
    driver's error. The catalog's psycopg reports a failed connect without a
    SQLSTATE; the vector store's asyncpg reports the connection limit as
    ``53300`` and a refused connection as a plain ``OSError``.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_unreachable_error(current):
            return True
        current = (
            getattr(current, "orig", None) or current.__cause__ or current.__context__
        )
    return False


def _is_unreachable_error(exc: BaseException) -> bool:
    sqlstate = getattr(exc, "sqlstate", None)
    if isinstance(sqlstate, str):
        return (
            sqlstate.startswith(_UNREACHABLE_SQLSTATE_CLASSES)
            or sqlstate in _UNREACHABLE_SQLSTATES
        )
    if isinstance(exc, psycopg.OperationalError):
        # No SQLSTATE: libpq failed to connect, or lost the connection.
        return True
    if getattr(exc, "connection_invalidated", False):
        return True
    if isinstance(exc, (ConnectionError, socket.gaierror)):
        return True
    # asyncio's create_connection, trying each address of a host, raises a
    # bare OSError listing every failure.
    return type(exc) is OSError and str(exc).startswith("Multiple exceptions:")
