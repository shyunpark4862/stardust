"""The plot tests read the recorded text cache frozen (tests/plot/scenes.py): no TeX runs, and a
string that is not recorded is an error naming it (run tools/plot_text.py). The tests marked
`tex` use real TeX in a fresh cache and belong to the deep tier."""

import pytest

from stardust.plot import _text

from .scenes import TEXT


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("tex"):
            item.add_marker(pytest.mark.deep)


def pytest_configure(config):
    config.addinivalue_line("markers", "tex: runs TeX Live (deep tier)")


@pytest.fixture(autouse=True)
def text_cache(request, tmp_path_factory):
    if request.node.get_closest_marker("tex"):
        with _text.cache(tmp_path_factory.mktemp("text")):
            yield
    else:
        with _text.cache(TEXT, frozen=True):
            yield
