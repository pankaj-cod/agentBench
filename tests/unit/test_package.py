from importlib.metadata import version

import agentbench


def test_version_matches_installed_metadata() -> None:
    assert agentbench.__version__ == version("agentbench")
