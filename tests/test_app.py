"""Smoke test: the Streamlit app renders and the propose -> approve flow works."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_and_approval_flow_works():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception

    propose = next(b for b in at.button if b.label.startswith("Propose plan"))
    propose.click().run()
    assert not at.exception
    assert any("sent to Approvals" in s.value for s in at.success)

    next(b for b in at.button if b.label == "Approve").click().run()
    assert not at.exception
    assert not [b for b in at.button if b.label == "Approve"]  # queue is empty again
