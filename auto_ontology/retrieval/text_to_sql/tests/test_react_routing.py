# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""With two databases and no target, a statement runs where its tables are (#585).

``seen_tables`` only grows. Routing ``run_sql`` and ``submit`` by it locked an
episode out of both databases as soon as one search showed a table from each,
and the refusal told the model to retrieve more, which could only grow it.
"""

from __future__ import annotations

import pytest

from auto_ontology.retrieval.text_to_sql.react import loop
from auto_ontology.retrieval.text_to_sql.react import run_sql as tool
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    SHOP_CATALOG,
    SHOP_SCRIPT,
    FakeCatalog,
    SQLiteConnector,
)

CRM_SCRIPT = """
CREATE TABLE customers (customer_id INTEGER, region TEXT);
CREATE TABLE tickets (ticket_id INTEGER, customer_id INTEGER);
INSERT INTO customers VALUES (1, 'EMEA');
INSERT INTO tickets VALUES (7, 1);
"""
CRM_CATALOG = {
    "customers": ["customer_id", "region"],
    "tickets": ["ticket_id", "customer_id"],
}


def _seen(database: str, name: str) -> dict:
    return {
        "id": f"{database}.{name}",
        "name": name,
        "schema_name": None,
        "database_name": database,
    }


@pytest.fixture
def two_databases(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ToolContext, SQLiteConnector, SQLiteConnector]:
    """An episode that has seen tables from both ``shop`` and ``crm``."""
    shop = SQLiteConnector(SHOP_SCRIPT)
    crm = SQLiteConnector(CRM_SCRIPT)
    crm.database_name = "crm"
    catalogs = {"shop": FakeCatalog(SHOP_CATALOG), "crm": FakeCatalog(CRM_CATALOG)}
    monkeypatch.setattr(
        tool, "catalog_for", lambda ctx, connector: catalogs[connector.database_name]
    )
    ctx = ToolContext(retriever=None, connectors=[shop, crm])
    for database, names in (("shop", SHOP_CATALOG), ("crm", CRM_CATALOG)):
        for name in names:
            ctx.seen_tables[f"{database}.{name}"] = _seen(database, name)
            ctx.show_table(None, name)
    return ctx, shop, crm


def test_a_statement_runs_on_the_database_holding_its_tables(two_databases) -> None:
    ctx, shop, crm = two_databases

    assert tool.run_sql(ctx, "SELECT SUM(amount) AS total FROM orders").startswith(
        "1 rows"
    )
    assert tool.run_sql(ctx, "SELECT ticket_id FROM tickets").startswith("1 rows")

    assert shop.executed and "orders" in shop.executed[0]
    assert crm.executed and "tickets" in crm.executed[0]


def test_a_shared_table_name_goes_with_the_other_tables_named(two_databases) -> None:
    ctx, shop, crm = two_databases

    text = tool.run_sql(
        ctx,
        "SELECT c.status, SUM(o.amount) AS revenue FROM orders o "
        "JOIN customers c ON c.customer_id = o.cust_ref GROUP BY c.status",
    )

    assert not text.startswith("Error"), text
    assert len(shop.executed) == 1
    assert crm.executed == []


def test_a_statement_across_two_databases_is_refused(two_databases) -> None:
    ctx, shop, crm = two_databases

    text = tool.run_sql(
        ctx,
        "SELECT o.amount FROM orders o JOIN tickets t ON t.customer_id = o.cust_ref",
    )

    assert text.startswith("Error: this query names tables from more than one")
    assert "crm: tickets" in text and "shop: orders" in text
    assert "Retrieve the tables" not in text
    assert shop.executed == crm.executed == []


def test_a_statement_every_database_could_mean_is_refused(two_databases) -> None:
    ctx, shop, crm = two_databases

    text = tool.run_sql(ctx, "SELECT customer_id FROM customers")

    assert text.startswith("Error: every table this query names is in more than")
    assert "(crm, shop)" in text
    assert shop.executed == crm.executed == []


def test_a_cte_is_not_taken_for_a_table(two_databases) -> None:
    ctx, shop, _ = two_databases

    text = tool.run_sql(
        ctx,
        "WITH customers AS (SELECT cust_ref, amount FROM orders) "
        "SELECT SUM(amount) AS total FROM customers",
    )

    assert text.startswith("1 rows"), text
    assert len(shop.executed) == 1


def test_a_statement_naming_no_seen_table_says_the_databases_differ(
    two_databases,
) -> None:
    ctx, _, _ = two_databases

    text = tool.run_sql(ctx, "SELECT 1 AS one")

    assert text.startswith("Error: the tables retrieved so far come from more than")
    assert "(crm, shop)" in text


def test_submit_routes_by_the_statement_too(two_databases) -> None:
    ctx, _, crm = two_databases

    observation, submission = loop.submit(ctx, "SELECT ticket_id FROM tickets")

    assert submission is not None, observation
    assert crm.executed


def test_the_target_database_still_wins(two_databases) -> None:
    ctx, shop, _ = two_databases
    ctx.database_name = "shop"

    assert tool.connector_for(ctx, "SELECT ticket_id FROM tickets") is shop


def test_one_connector_still_takes_every_statement() -> None:
    shop = SQLiteConnector(SHOP_SCRIPT)
    ctx = ToolContext(retriever=None, connectors=[shop])
    ctx.seen_tables["crm.tickets"] = _seen("crm", "tickets")

    assert tool.connector_for(ctx, "SELECT ticket_id FROM tickets") is shop
