import os

import pytest

from bsi.sandbox import SandboxPool


@pytest.fixture(scope="session")
def pool():
    with SandboxPool(n_workers=2) as p:
        yield p


def pytest_collection_modifyitems(config, items):
    if os.environ.get("BSI_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live API test; set BSI_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
