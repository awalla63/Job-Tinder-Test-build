"""Response shapes: SQL column aliases must match the Pydantic schemas and the
frontend's TypeScript types.

These three places describe the same JSON independently -- the SQL `select` list, the
schema in schemas.py, and apps/web/src/lib/api.ts -- and nothing forces them to agree.
This caught a real bug during the live-database smoke test: list_matches() selected
`j.title` with no alias, but both MatchOut.job_title and the frontend's MatchRow.job_title
expected the key `job_title`. The route had no response_model, so FastAPI serialised the
dict as-is and nothing short of actually calling it with real data surfaced the mismatch
-- the existing auth/match-rule tests mocked the cursor and never looked at column names.
"""
from __future__ import annotations

import inspect
import re

from services.api.routers import matching
from services.api.schemas import MatchOut

# Columns the frontend's MatchRow type (apps/web/src/lib/api.ts) expects by name.
# Kept here rather than imported since there is no shared source of truth across the
# Python/TypeScript boundary -- this test *is* that source of truth until one exists.
FRONTEND_MATCH_ROW_FIELDS = {
    "id", "job_id", "job_title", "company_name", "application_url",
    "source", "counterpart_name", "created_at",
}


def _select_list_aliases(source: str) -> set[str]:
    """Pull the output column names from a `select ... from` block.

    Handles both `column` and `expr as alias` forms, which is all this query uses.
    """
    select_block = re.search(r"select\s+(.*?)\s+from\s", source, re.S).group(1)
    names = set()
    for item in select_block.split(","):
        item = item.strip()
        m = re.search(r"\bas\s+(\w+)$", item, re.I)
        if m:
            names.add(m.group(1))
        else:
            # Strip a `::type` cast (e.g. `j.source::text`) before taking the bare
            # column name -- Postgres' implicit output name ignores the cast too.
            bare = item.split("::")[0].strip()
            names.add(bare.split(".")[-1])
    return names


def test_matches_query_columns_match_the_match_out_schema():
    source = inspect.getsource(matching.list_matches)
    aliases = _select_list_aliases(source)
    schema_fields = set(MatchOut.model_fields.keys())
    # created_at and job_id are reshaped after the query (isoformat/str cast) rather
    # than aliased in SQL, so they're not expected to appear verbatim here.
    expected = schema_fields - {"created_at", "job_id"}
    missing = expected - aliases
    assert not missing, (
        f"MatchOut declares {missing} but the SQL in list_matches() does not select "
        f"a column with that exact name. Alias it with `as <field_name>`."
    )


def test_matches_query_columns_match_the_frontend_type():
    source = inspect.getsource(matching.list_matches)
    aliases = _select_list_aliases(source)
    expected = FRONTEND_MATCH_ROW_FIELDS - {"id", "job_id", "created_at"}
    missing = expected - aliases
    assert not missing, (
        f"apps/web/src/lib/api.ts's MatchRow expects {missing} but the SQL in "
        f"list_matches() does not select a column with that exact name."
    )
