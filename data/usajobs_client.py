"""USAJOBS Search API connector.

Key from https://developer.usajobs.gov/apirequest/. The User-Agent header must be the
email address you registered with -- requests are rejected otherwise.

Terms of service (https://developer.usajobs.gov/guides/terms-of-service) permit storing
and redisplaying announcements, but require that we:
  * do not alter data values,
  * credit USAJOBS as the source,
  * send users to the official announcement to apply (we must not collect applications),
  * disclose that this app is not endorsed by OPM.
The UI honours these by rendering a source attribution line on USAJOBS cards and linking
`application_url` straight to PositionURI. OPM guarantees no archival availability, so
refreshing stale rows is our responsibility.

This module intentionally mirrors the shape a Greenhouse/Lever/Ashby connector would
take -- fetch, map to the `jobs` columns, upsert on (source, external_id) -- so adding
those later is a matter of writing a new `fetch` + `map_record` pair.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterator

import httpx

BASE_URL = "https://data.usajobs.gov/api/search"
MAX_RESULTS_PER_PAGE = 500          # documented hard cap
MAX_RESULTS_PER_QUERY = 10_000      # documented pagination ceiling
THROTTLE_SECONDS = 0.5              # no published rate limit; self-throttle as a courtesy


class USAJobsError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    api_key = os.environ.get("USAJOBS_API_KEY")
    user_agent = os.environ.get("USAJOBS_USER_AGENT")
    if not api_key or not user_agent:
        raise USAJobsError(
            "USAJOBS_API_KEY and USAJOBS_USER_AGENT must be set in .env. "
            "Request a key at https://developer.usajobs.gov/apirequest/ -- the "
            "User-Agent must be the email address you registered with."
        )
    return {
        "Host": "data.usajobs.gov",
        "User-Agent": user_agent,
        "Authorization-Key": api_key,
    }


def search(
    keyword: str,
    *,
    location: str | None = None,
    days_posted: int | None = 30,
    remote_only: bool = False,
    results_per_page: int = 250,
    max_results: int = 1000,
    client: httpx.Client | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw SearchResultItems for a keyword, paging until exhausted."""
    results_per_page = min(results_per_page, MAX_RESULTS_PER_PAGE)
    max_results = min(max_results, MAX_RESULTS_PER_QUERY)

    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        fetched = 0
        page = 1
        while fetched < max_results:
            params: dict[str, Any] = {
                "Keyword": keyword,
                "Page": page,
                "ResultsPerPage": results_per_page,
            }
            if location:
                params["LocationName"] = location
            if days_posted is not None:
                params["DatePosted"] = max(0, min(60, days_posted))  # API accepts 0-60
            if remote_only:
                params["RemoteIndicator"] = "True"

            response = client.get(BASE_URL, headers=_headers(), params=params)
            if response.status_code != 200:
                raise USAJobsError(f"USAJOBS returned {response.status_code}: {response.text[:300]}")

            payload = response.json().get("SearchResult", {})
            items = payload.get("SearchResultItems", []) or []
            if not items:
                return

            for item in items:
                yield item
                fetched += 1
                if fetched >= max_results:
                    return

            total_pages = int(payload.get("UserArea", {}).get("NumberOfPages", page) or page)
            if page >= total_pages:
                return
            page += 1
            time.sleep(THROTTLE_SECONDS)
    finally:
        if owns_client:
            client.close()


def _first(seq: Any) -> Any:
    return seq[0] if isinstance(seq, list) and seq else None


# RateIntervalCode -> the pay_period vocabulary the LinkedIn corpus already uses,
# so both sources populate the column consistently.
RATE_INTERVAL = {
    "PA": "YEARLY", "PH": "HOURLY", "PD": "DAILY",
    "PW": "WEEKLY", "BW": "BIWEEKLY", "PM": "MONTHLY",
}

# Annualisation factors for normalized_salary, which downstream features assume is
# a yearly figure. Federal postings are overwhelmingly PA, but hourly appears.
ANNUALISE = {"YEARLY": 1, "HOURLY": 2080, "DAILY": 260, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12}


def map_record(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map one SearchResultItem onto the `jobs` columns."""
    descriptor = item.get("MatchedObjectDescriptor") or {}
    title = (descriptor.get("PositionTitle") or "").strip()
    summary = ((descriptor.get("UserArea") or {}).get("Details") or {}).get("JobSummary") or ""
    if not title or not summary:
        return None

    remuneration = _first(descriptor.get("PositionRemuneration")) or {}
    interval = RATE_INTERVAL.get(remuneration.get("RateIntervalCode"))
    salary_min = remuneration.get("MinimumRange")
    salary_max = remuneration.get("MaximumRange")

    normalized_salary = None
    try:
        values = [float(v) for v in (salary_min, salary_max) if v not in (None, "")]
        if values:
            normalized_salary = (sum(values) / len(values)) * ANNUALISE.get(interval, 1)
    except (TypeError, ValueError):
        normalized_salary = None

    location_obj = _first(descriptor.get("PositionLocation")) or {}

    return {
        "external_id": str(item.get("MatchedObjectId") or descriptor.get("PositionID") or "").strip(),
        "company_name": descriptor.get("OrganizationName") or descriptor.get("DepartmentName") or "",
        "title": title,
        "description": summary,
        "location": descriptor.get("PositionLocationDisplay"),
        "state": location_obj.get("CountrySubDivisionCode"),
        "normalized_salary": normalized_salary,
        "pay_period": interval,
        "formatted_work_type": "Full-time",
        "formatted_experience_level": None,  # USAJOBS grades don't map cleanly; leave unknown
        "remote_allowed": bool(((descriptor.get("UserArea") or {}).get("Details") or {})
                               .get("TeleworkEligible") or False),
        # PositionURI is the official announcement -- ToS requires sending users there.
        "application_url": descriptor.get("PositionURI") or _first(descriptor.get("ApplyURI")),
        "posted_at": descriptor.get("PublicationStartDate") or None,
    }
