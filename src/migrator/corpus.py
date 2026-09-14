"""Generates a synthetic legacy codebase to migrate.

Why generate instead of vendoring a real repo: the benchmark needs a *known*
distribution of change shapes so the deterministic pass and the clustering
pass can be scored against ground truth. A real repo gives you realism but no
answer key. This generator gives both -- the code it emits is ordinary,
slightly repetitive service code of the kind that actually accumulates in a
company codebase, and every call site is tagged with the shape it belongs to.

The migration modelled is ``requests`` -> ``httpx``, chosen because it has a
genuinely mixed character:

* a large mechanical core (``requests.get`` -> ``httpx.get``),
* a renamed-keyword tier (``allow_redirects`` -> ``follow_redirects``),
* and a residue that needs a judgement call, because the two libraries have
  different *default* behaviour. ``requests`` defaults to no timeout at all;
  ``httpx`` defaults to five seconds. A call site with no explicit timeout
  therefore cannot be migrated mechanically without changing behaviour, and
  what the right timeout is depends on what the call is doing.

That residue is the whole point. It is the part a codemod cannot do and a
model can, and it is small -- which is exactly why sending every file to the
model is such a bad trade.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Shape(str, Enum):
    """The change shape a call site belongs to.

    ``MECHANICAL_*`` shapes are fully resolvable by the deterministic pass.
    ``AMBIGUOUS_*`` shapes are not, and are what the model is for.
    """

    MECHANICAL_CALL = "mechanical_call"
    MECHANICAL_REDIRECT = "mechanical_redirect"
    MECHANICAL_SESSION = "mechanical_session"
    AMBIGUOUS_TIMEOUT = "ambiguous_timeout"
    AMBIGUOUS_RETRY = "ambiguous_retry"
    AMBIGUOUS_STATUS = "ambiguous_status"

    #: Generator-only. The scanner cannot distinguish an interactive
    #: missing-timeout call from a batch one -- to the AST they are the same
    #: shape, and that is the correct level of discrimination for a *rule*.
    #: The distinction lives in the cluster fingerprint instead, which carries
    #: the enclosing function's character. This member therefore never appears
    #: in scanner output, and ``_SITES_PER_TEMPLATE`` maps it back onto
    #: ``AMBIGUOUS_TIMEOUT`` so ground truth stays comparable.
    AMBIGUOUS_TIMEOUT_INTERACTIVE = "ambiguous_timeout_interactive"

    @property
    def is_mechanical(self) -> bool:
        return self.name.startswith("MECHANICAL")


@dataclass(frozen=True)
class SourceFile:
    path: str
    text: str
    #: Ground truth: how many sites of each shape this file contains.
    shapes: Dict[Shape, int]

    @property
    def touched(self) -> bool:
        return bool(self.shapes)

    @property
    def needs_model(self) -> bool:
        return any(not shape.is_mechanical for shape in self.shapes)


_PACKAGES = [
    "billing",
    "accounts",
    "catalog",
    "notifications",
    "shipping",
    "analytics",
    "auth",
    "search",
]

_NOUNS = [
    "invoice",
    "customer",
    "product",
    "shipment",
    "webhook",
    "session",
    "receipt",
    "tenant",
    "subscription",
    "ledger",
    "quote",
    "refund",
]

_SERVICES = [
    "billing-api",
    "identity",
    "warehouse",
    "pricing",
    "partner-gateway",
    "reporting",
]


def _header(module: str, service: str) -> str:
    return f'''"""Client helpers for the {service} service.

Extracted from the monolith during the 2019 service split. Do not add new
callers here -- use the generated client in ``clients/{service.replace("-", "_")}``.
"""

import logging
import os
import time

import requests

log = logging.getLogger(__name__)

BASE_URL = os.environ.get("{service.upper().replace("-", "_")}_URL", "https://{service}.internal")
DEFAULT_HEADERS = {{"accept": "application/json", "user-agent": "{module}/1.0"}}
'''


def _mechanical_call(noun: str, rng: random.Random) -> str:
    verb = rng.choice(["get", "post", "put"])
    timeout = rng.choice([5, 10, 15, 30])
    if verb == "get":
        call = f'requests.get(f"{{BASE_URL}}/{noun}s/{{{noun}_id}}", headers=DEFAULT_HEADERS, timeout={timeout})'
    else:
        call = (
            f'requests.{verb}(f"{{BASE_URL}}/{noun}s", json=payload, '
            f"headers=DEFAULT_HEADERS, timeout={timeout})"
        )
    args = f"{noun}_id" if verb == "get" else "payload"
    return f'''

def fetch_{noun}({args}):
    """Return the {noun} record, or None when the service 404s."""
    response = {call}
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()
'''


def _mechanical_redirect(noun: str, rng: random.Random) -> str:
    flag = rng.choice(["False", "True"])
    return f'''

def resolve_{noun}_url({noun}_id):
    """Follow the short-link indirection and return the canonical URL."""
    response = requests.head(
        f"{{BASE_URL}}/s/{{{noun}_id}}",
        allow_redirects={flag},
        timeout=5,
    )
    return response.headers.get("location", response.url)
'''


def _mechanical_session(noun: str, rng: random.Random) -> str:
    return f'''

def bulk_load_{noun}s({noun}_ids):
    """Load many {noun}s over one connection."""
    results = []
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    for {noun}_id in {noun}_ids:
        response = session.get(f"{{BASE_URL}}/{noun}s/{{{noun}_id}}", timeout=10)
        if response.ok:
            results.append(response.json())
    session.close()
    return results
'''


def _ambiguous_timeout(noun: str, rng: random.Random) -> str:
    """A call with no timeout at all.

    Under ``requests`` this blocks forever; under ``httpx`` it would silently
    become a 5 second timeout. Neither "keep blocking forever" nor "accept the
    new default" is automatically right, and the correct answer depends on
    whether this is an interactive path or a batch job -- which is visible in
    the surrounding code, not in the call itself.
    """
    flavour = rng.choice(["report", "export", "sync"])
    return f'''

def {flavour}_{noun}s(window_days):
    """Pull the full {noun} {flavour} for the given window.

    Runs from the nightly cron. The {flavour} is large and the service streams
    it slowly, so this intentionally has no timeout today.
    """
    response = requests.get(
        f"{{BASE_URL}}/{noun}s/{flavour}",
        params={{"window": window_days}},
        headers=DEFAULT_HEADERS,
    )
    response.raise_for_status()
    return response.json()["rows"]
'''


def _ambiguous_retry(noun: str, rng: random.Random) -> str:
    """Hand-rolled retry around requests-specific exception types.

    ``httpx`` does not have ``requests.exceptions.ConnectionError``; the
    nearest equivalent is ``httpx.ConnectError``, but the exception *hierarchy*
    differs enough that a blind rename changes which failures get retried.
    """
    attempts = rng.choice([3, 4, 5])
    return f'''

def push_{noun}(payload):
    """Send the {noun} upstream, retrying transient network failures."""
    for attempt in range({attempts}):
        try:
            response = requests.post(
                f"{{BASE_URL}}/{noun}s",
                json=payload,
                headers=DEFAULT_HEADERS,
                timeout=20,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            log.warning("connection failed for {noun}, attempt %s", attempt + 1)
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            log.warning("timeout pushing {noun}, attempt %s", attempt + 1)
            time.sleep(2 ** attempt)
    raise RuntimeError("could not push {noun} after {attempts} attempts")
'''


def _ambiguous_status(noun: str, rng: random.Random) -> str:
    """``raise_for_status`` caught as ``requests.exceptions.HTTPError``.

    In ``httpx`` this becomes ``httpx.HTTPStatusError``, and crucially the
    attribute used to read the status code off the caught exception differs,
    so the handler body needs editing too -- not just the ``except`` clause.
    """
    return f'''

def delete_{noun}({noun}_id):
    """Delete the {noun}. Treats 'already gone' as success."""
    try:
        response = requests.delete(
            f"{{BASE_URL}}/{noun}s/{{{noun}_id}}",
            headers=DEFAULT_HEADERS,
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return False
        raise
    return True
'''


def _ambiguous_timeout_interactive(noun: str, rng: random.Random) -> str:
    """The same missing-timeout shape, on an interactive path.

    This template exists because the batch variant alone made the corpus a
    liar. If every timeout-less call is a nightly job, then "use 120 seconds"
    is always right, the batch-vs-interactive judgement is never actually
    tested, and the clusterer never has to prove it can tell the two apart.

    Real codebases are not like that. The request-path call with no timeout is
    both common and the more dangerous of the two, because inheriting a long
    timeout here pins a worker thread instead of failing fast. The correct
    answers for the two shapes differ by more than an order of magnitude, so
    anything that merges them is actively harmful -- which makes this the
    sharpest available test of whether the fingerprint carries enough context.
    """
    return f'''

def load_{noun}_for_request({noun}_id):
    """Fetch the {noun} to render the detail page. Called per page view."""
    response = requests.get(
        f"{{BASE_URL}}/{noun}s/{{{noun}_id}}",
        headers=DEFAULT_HEADERS,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()
'''


_BUILDERS = {
    Shape.MECHANICAL_CALL: _mechanical_call,
    Shape.MECHANICAL_REDIRECT: _mechanical_redirect,
    Shape.MECHANICAL_SESSION: _mechanical_session,
    Shape.AMBIGUOUS_TIMEOUT: _ambiguous_timeout,
    Shape.AMBIGUOUS_TIMEOUT_INTERACTIVE: _ambiguous_timeout_interactive,
    Shape.AMBIGUOUS_RETRY: _ambiguous_retry,
    Shape.AMBIGUOUS_STATUS: _ambiguous_status,
}

#: How many migration *sites* each template actually emits, by shape.
#:
#: This is not the same as "one template, one site", and conflating the two
#: quietly corrupts the answer key. The retry template, for example, contains a
#: perfectly mechanical ``requests.post(..., timeout=20)`` alongside its two
#: ``except`` clauses, so it contributes one mechanical site and two ambiguous
#: ones. Counting it as a single "ambiguous" unit would understate the
#: deterministic pass and overstate the model's share of the work -- an error
#: in the flattering direction, which is the kind worth being strict about.
_SITES_PER_TEMPLATE = {
    Shape.MECHANICAL_CALL: {Shape.MECHANICAL_CALL: 1},
    Shape.MECHANICAL_REDIRECT: {Shape.MECHANICAL_REDIRECT: 1},
    Shape.MECHANICAL_SESSION: {Shape.MECHANICAL_SESSION: 1},
    Shape.AMBIGUOUS_TIMEOUT: {Shape.AMBIGUOUS_TIMEOUT: 1},
    # Same shape to the scanner; only the surrounding context differs.
    Shape.AMBIGUOUS_TIMEOUT_INTERACTIVE: {Shape.AMBIGUOUS_TIMEOUT: 1},
    # one mechanical POST + two requests-specific except clauses
    Shape.AMBIGUOUS_RETRY: {Shape.MECHANICAL_CALL: 1, Shape.AMBIGUOUS_RETRY: 2},
    # one mechanical DELETE + one HTTPError clause
    Shape.AMBIGUOUS_STATUS: {Shape.MECHANICAL_CALL: 1, Shape.AMBIGUOUS_STATUS: 1},
}

#: Shape distribution, weighted to match what real migrations look like: the
#: long tail of judgement calls is real but small. If you flip these weights so
#: most sites are ambiguous, the deterministic pass stops being the dominant
#: lever -- which is the honest answer for a migration with no mechanical core.
_WEIGHTS = {
    Shape.MECHANICAL_CALL: 44,
    Shape.MECHANICAL_REDIRECT: 10,
    Shape.MECHANICAL_SESSION: 12,
    Shape.AMBIGUOUS_TIMEOUT: 9,
    Shape.AMBIGUOUS_TIMEOUT_INTERACTIVE: 9,
    Shape.AMBIGUOUS_RETRY: 10,
    Shape.AMBIGUOUS_STATUS: 8,
}

_UNTOUCHED_TEMPLATE = '''"""Pure helpers for the {package} package.

No I/O here by design -- this module is imported by the request path and must
stay cheap.
"""

from decimal import Decimal


def normalise_{noun}(raw):
    """Coerce a raw {noun} dict into the internal shape."""
    return {{
        "id": str(raw["id"]),
        "amount": Decimal(str(raw.get("amount", "0"))),
        "currency": raw.get("currency", "USD").upper(),
        "active": bool(raw.get("active", True)),
    }}


def summarise_{noun}s(rows):
    """Total the amounts, grouped by currency."""
    totals = {{}}
    for row in rows:
        totals.setdefault(row["currency"], Decimal("0"))
        totals[row["currency"]] += row["amount"]
    return totals
'''


def build_corpus(
    file_count: int = 120,
    untouched_ratio: float = 0.30,
    seed: int = 20240914,
) -> List[SourceFile]:
    """Return a deterministic synthetic codebase.

    ``untouched_ratio`` matters more than it looks. Roughly a third of any real
    codebase does not import the library being migrated at all, and a naive
    agent that walks every file pays for those too. Setting this to zero would
    flatter the optimised pipeline by removing a saving it genuinely makes.
    """
    rng = random.Random(seed)
    shapes = list(_WEIGHTS)
    weights = [_WEIGHTS[s] for s in shapes]
    files: List[SourceFile] = []

    for index in range(file_count):
        package = _PACKAGES[index % len(_PACKAGES)]
        noun = _NOUNS[index % len(_NOUNS)]

        if rng.random() < untouched_ratio:
            files.append(
                SourceFile(
                    path=f"src/{package}/{noun}_utils_{index}.py",
                    text=_UNTOUCHED_TEMPLATE.format(package=package, noun=noun),
                    shapes={},
                )
            )
            continue

        service = rng.choice(_SERVICES)
        module = f"{package}.{noun}_client_{index}"
        parts = [_header(module, service)]
        counts: Dict[Shape, int] = {}

        for _ in range(rng.choice([1, 2, 2, 3, 3, 4])):
            shape = rng.choices(shapes, weights=weights, k=1)[0]
            local_noun = rng.choice(_NOUNS)
            parts.append(_BUILDERS[shape](local_noun, rng))
            for emitted, n in _SITES_PER_TEMPLATE[shape].items():
                counts[emitted] = counts.get(emitted, 0) + n

        files.append(
            SourceFile(
                path=f"src/{package}/{noun}_client_{index}.py",
                text="".join(parts),
                shapes=counts,
            )
        )

    return files


def corpus_stats(files: List[SourceFile]) -> Dict[str, int]:
    """Ground-truth summary, used by tests and by the benchmark header."""
    totals: Dict[str, int] = {
        "files": len(files),
        "files_touched": sum(1 for f in files if f.touched),
        "files_needing_model": sum(1 for f in files if f.needs_model),
        "sites": 0,
        "mechanical_sites": 0,
        "ambiguous_sites": 0,
    }
    for source in files:
        for shape, n in source.shapes.items():
            totals["sites"] += n
            key = "mechanical_sites" if shape.is_mechanical else "ambiguous_sites"
            totals[key] += n
    return totals
