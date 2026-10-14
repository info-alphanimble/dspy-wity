import re
from pathlib import Path

import dspy_wity
from dspy_wity import __all__, __version__


def test_everything_in_all_imports() -> None:
    for name in __all__:
        assert getattr(dspy_wity, name) is not None


def test_version_matches_pyproject() -> None:
    # A regex, not tomllib: tomllib needs Python 3.11 and we support 3.10.
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    match = re.search(r'^version = "(.+)"$', pyproject, re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)


def test_dspy_has_the_decision_types() -> None:
    from dspy.experimental import Choice, Noul, ReAnchor, Score  # noqa: F401
