"""verify-on-stop must not nudge while this turn's background delegation is still running (#53828).

The delegation's result re-enters the session as a new turn. A nudge before that cannot be
satisfied, so the model restates the answer the user already sees, once per nudge attempt.
"""

import time
from types import SimpleNamespace

import pytest

import agent.turn_stop_gates as gates
from tools import async_delegation as ad

_SESSION = "verify-pending-delegation"


@pytest.fixture
def gate_env(monkeypatch):
    """verify-on-stop enabled and always demanding evidence; every other gate silent."""
    import agent.verification_stop as vs

    monkeypatch.setattr(vs, "verify_on_stop_enabled", lambda: True)
    monkeypatch.setattr(vs, "build_verify_on_stop_nudge", lambda **_kw: "VERIFY NOW")
    monkeypatch.setattr(gates, "_late_lsp_verdicts", lambda _agent: None)
    monkeypatch.setattr(gates, "_pre_verify_nudge", lambda *_a: None)
    monkeypatch.setattr(gates, "_kanban_stop_nudge", lambda *_a: None)
    monkeypatch.setattr(gates, "_append_interim_answer", lambda *_a: None)
    monkeypatch.setattr(ad, "_records", {})
    return ad._records


def _agent(turn_started: float) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=_SESSION, _current_turn_timestamp=turn_started,
        _turn_file_mutation_paths={"/repo/app.py"}, _session_messages=None,
        _interim_content_was_streamed=lambda _text: True,
    )


def _delegation(records, *, status="running", dispatched_at, parent=_SESSION):
    records[f"deleg_{len(records)}"] = {
        "status": status, "dispatched_at": dispatched_at, "parent_session_id": parent,
        "session_key": "", "origin_ui_session_id": ""}


def _stop(agent):
    return gates.apply_stop_gates(
        agent, {"role": "assistant", "content": "answer"}, final_response="answer", messages=[],
        conversation_history=[], pending_verification_response=None,
        pending_verification_response_previewed=None)


class TestVerifyOnStopWithPendingDelegation:
    def test_running_delegation_from_this_turn_ends_the_turn_without_a_nudge(self, gate_env):
        started = time.time()
        _delegation(gate_env, dispatched_at=started + 1)
        verdict = _stop(_agent(started))
        assert verdict.continue_turn is False
        assert verdict.final_response == "answer"

    @pytest.mark.parametrize("status", ["stalling", "finalizing"])
    def test_other_live_states_also_count(self, gate_env, status):
        started = time.time()
        _delegation(gate_env, status=status, dispatched_at=started + 1)
        assert _stop(_agent(started)).continue_turn is False

    def test_no_delegation_still_nudges(self, gate_env):
        verdict = _stop(_agent(time.time()))
        assert verdict.continue_turn is True
        assert verdict.final_response is None

    def test_finished_delegation_still_nudges(self, gate_env):
        started = time.time()
        _delegation(gate_env, status="completed", dispatched_at=started + 1)
        assert _stop(_agent(started)).continue_turn is True

    def test_delegation_from_an_earlier_turn_still_nudges(self, gate_env):
        """An unrelated long-running subagent must not silence the gate for this turn's edits."""
        started = time.time()
        _delegation(gate_env, dispatched_at=started - 60)
        assert _stop(_agent(started)).continue_turn is True

    def test_another_sessions_delegation_still_nudges(self, gate_env):
        started = time.time()
        _delegation(gate_env, dispatched_at=started + 1, parent="someone-else")
        assert _stop(_agent(started)).continue_turn is True
