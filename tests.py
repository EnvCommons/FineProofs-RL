"""Offline tests for the submit_proof grading path, with a scripted grader client.

The server loads fineproofs_train.parquet from the working directory at import, so
the tests import it from a temporary directory holding a one-task stand-in.

Run: uv run --no-project --with-requirements requirements.txt --with pytest python -m pytest tests.py -q
"""
import asyncio
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def mod(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    pd.DataFrame([{"problem": "Prove that 1 = 1.", "source": "test", "rubrics": "HIDDEN RUBRIC"}]).to_parquet(
        tmp / "fineproofs_train.parquet")
    cwd = os.getcwd()
    sys.path.insert(0, str(Path(__file__).parent))
    os.chdir(tmp)
    try:
        sys.modules.pop("server", None)
        return importlib.import_module("server")
    finally:
        os.chdir(cwd)


class ScriptedClient:
    """Stands in for openai.AsyncClient; returns the scripted replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))])


def _submit(mod, replies: list[str]):
    env = mod.FineProofsRL(task_spec=mod.test_tasks[0], secrets={"openai_api_key": "test"})
    env.client = ScriptedClient(replies)
    return env, asyncio.run(env.submit_proof(mod.ProofInput(proof="1 = 1.")))


def test_scored_reply(mod):
    env, out = _submit(mod, ["Looks right.\nScore: 7"])
    assert out.reward == 1.0 and out.finished is True and env.client.calls == 1


def test_scoreless_reply_is_resampled(mod):
    env, out = _submit(mod, ["", "no digits here", "Score: 3"])
    assert out.reward == pytest.approx(3 / 7) and out.finished is True and env.client.calls == 3


def test_scoreless_on_every_attempt_raises_and_keeps_attempt(mod):
    env = mod.FineProofsRL(task_spec=mod.test_tasks[0], secrets={"openai_api_key": "test"})
    env.client = ScriptedClient([""] * mod.GRADER_SCORE_ATTEMPTS)
    with pytest.raises(RuntimeError, match="no score") as exc:
        asyncio.run(env.submit_proof(mod.ProofInput(proof="1 = 1.")))
    assert "HIDDEN RUBRIC" not in str(exc.value)
    assert env.client.calls == mod.GRADER_SCORE_ATTEMPTS and env.submitted == 0
