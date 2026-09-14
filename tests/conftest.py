"""Shared fixtures.

``src`` goes on the path here rather than via an editable install so the suite
runs from a clean checkout with nothing but pytest available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from migrator.config import DEFAULT  # noqa: E402
from migrator.corpus import build_corpus  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    return build_corpus()


@pytest.fixture(scope="session")
def small_corpus():
    return build_corpus(file_count=24, seed=7)


@pytest.fixture
def settings():
    return DEFAULT


MECHANICAL_SOURCE = '''\
import requests

BASE_URL = "https://example.internal"


def fetch(item_id):
    """Fetch an item.

    Historically this used requests.get with no timeout, which was a bug.
    """
    # requests.post here would be wrong; we only read.
    response = requests.get(
        f"{BASE_URL}/items/{item_id}",
        allow_redirects=False,
        timeout=10,
    )
    return response.json()
'''

AMBIGUOUS_SOURCE = '''\
import requests


def export_ledger(window):
    """Nightly export. Streams slowly."""
    response = requests.get("https://x.internal/ledger", params={"w": window})
    response.raise_for_status()
    return response.json()


def push(payload):
    for attempt in range(3):
        try:
            return requests.post("https://x.internal/p", json=payload, timeout=20).json()
        except requests.exceptions.ConnectionError:
            continue
        except requests.exceptions.Timeout:
            continue
    raise RuntimeError("failed")
'''
