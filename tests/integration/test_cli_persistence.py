import io
import json

import pytest
from sqlalchemy import Engine

from agentbench.cli import main
from agentbench.persistence.db import DATABASE_URL_ENV

pytestmark = pytest.mark.postgres


def test_saved_run_can_be_shown_again(
    migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI opens its own connection, so this run is committed to the throwaway
    # session database (dropped when the test session ends).
    monkeypatch.setenv(DATABASE_URL_ENV, migrated_engine.url.render_as_string(hide_password=False))

    out = io.StringIO()
    assert main(["run", "What is the price of record B-07?", "--save", "--json"], out=out) == 0
    saved = json.loads(out.getvalue())

    shown = io.StringIO()
    assert main(["show", saved["run_id"], "--json"], out=shown) == 0
    assert json.loads(shown.getvalue()) == saved
