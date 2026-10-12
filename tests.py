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
from unittest.mock import patch

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
    """OpenReward backend stand-in; captures the exact provider request options."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.options = []
        self.closed = False

    async def create(self, **kwargs):
        self.calls += 1
        self.options.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            text, finish = reply
        else:
            text, finish = reply, "stop"
        return SimpleNamespace(text=text, finish_reason=finish)

    async def aclose(self):
        self.closed = True


def _env(mod, replies):
    backend = ScriptedClient(replies)
    with patch.object(mod, 'resolve_backend', return_value=backend) as resolve:
        env = mod.FineProofsRL(task_spec=mod.test_tasks[0], secrets={"openai_api_key": "test"})
        resolve.assert_called_once_with(provider='openai', secrets={'openai_api_key': 'test'})
    return env


def _submit(mod, replies):
    env = _env(mod, replies)
    return env, asyncio.run(env.submit_proof(mod.ProofInput(proof="1 = 1.")))


def test_scored_reply_and_exact_grader_options(mod):
    env, out = _submit(mod, ["Looks right.\nScore: 7"])
    assert out.reward == 1.0 and out.finished is True and env.client.calls == 1
    options = env.client.options[0]
    assert options['model'] == 'gpt-6-sol'
    assert options['reasoning_effort'] == 'medium'
    assert options['max_completion_tokens'] == 24000
    assert options['service_tier'] == 'default'
    assert 'temperature' not in options
    assert 'HIDDEN RUBRIC' in options['messages'][0]['content']
    assert out.metadata['grader'] == 'gpt-6-sol'
    assert out.metadata['grader_reasoning_effort'] == 'medium'
    assert 'HIDDEN RUBRIC' not in out.model_dump_json()
    assert 'Looks right.' not in out.model_dump_json()
    assert 'HIDDEN RUBRIC' not in env.get_prompt()[0].text


def test_scoreless_reply_is_resampled(mod):
    env, out = _submit(mod, ["", "no digits here", "Score: 3"])
    assert out.reward == pytest.approx(3 / 7) and out.finished is True and env.client.calls == 3


@pytest.mark.parametrize('finish', ['length', 'content_filter', None])
def test_incomplete_reply_with_score_is_not_rewarded(mod, finish):
    env, out = _submit(mod, [("Score: 7", finish), "Score: 2"])
    assert out.reward == pytest.approx(2 / 7) and env.client.calls == 2


def test_scoreless_on_every_attempt_raises_and_keeps_attempt(mod):
    env = _env(mod, [""] * mod.GRADER_SCORE_ATTEMPTS)
    with pytest.raises(RuntimeError, match="no score") as exc:
        asyncio.run(env.submit_proof(mod.ProofInput(proof="1 = 1.")))
    assert "HIDDEN RUBRIC" not in str(exc.value)
    assert env.client.calls == mod.GRADER_SCORE_ATTEMPTS and env.submitted == 0


def test_provider_failure_is_not_zero_or_a_consumed_submission(mod):
    env = _env(mod, [RuntimeError('Provider unavailable')])
    with pytest.raises(RuntimeError, match='Grading failed'):
        asyncio.run(env.submit_proof(mod.ProofInput(proof='1 = 1.')))
    assert env.submitted == 0 and env.client.calls == 1


def test_repeat_submission_penalty_and_backend_cleanup(mod):
    env, first = _submit(mod, ['Score: 7'])
    async def finish():
        repeated = await env.submit_proof(mod.ProofInput(proof='Another proof'))
        assert repeated.reward == -.1 and repeated.finished
        assert first.reward == 1 and env.client.calls == 1
        await env.teardown()
        assert env.client.closed
    asyncio.run(finish())


def test_backend_requires_session_secret_even_when_env_key_exists(mod, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'policy-key')
    monkeypatch.setenv('CHAT_API_KEY', 'environment-key')
    for secrets in ({}, {'openai_api_key': ''}, {'openai_api_key': '  '}):
        with pytest.raises(ValueError, match='key required in secrets'):
            mod.FineProofsRL(mod.test_tasks[0], secrets=secrets)


def test_backend_preserves_gateway_and_dedicated_judge_routing(mod, monkeypatch):
    for name in ('CHAT_BASE_URL', 'CHAT_API_KEY', 'CHAT_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://policy-gateway.example/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'policy-key')
    env = mod.FineProofsRL(mod.test_tasks[0], secrets={'openai_api_key': 'session-key'})
    assert env.client.config.base_url == 'https://policy-gateway.example/v1'
    assert env.client.config.api_key == 'session-key'
    asyncio.run(env.teardown())

    env = mod.FineProofsRL(mod.test_tasks[0], secrets={'chat_api_key': 'judge-key'})
    assert env.client.config.base_url == 'https://api.openai.com/v1'
    assert env.client.config.api_key == 'judge-key'
    asyncio.run(env.teardown())

    monkeypatch.setenv('CHAT_BASE_URL', 'https://judge-gateway.example/v1')
    env = mod.FineProofsRL(mod.test_tasks[0], secrets={'openai_api_key': 'session-key'})
    assert env.client.config.base_url == 'https://judge-gateway.example/v1'
    asyncio.run(env.teardown())


def test_real_http_server_lifecycle_and_reward(mod, monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from openreward.environments import Server
    from sse_starlette.sse import AppStatus
    monkeypatch.setattr(AppStatus, 'should_exit_event', None)
    backend = ScriptedClient(['Analysis: HIDDEN RUBRIC checkpoints.\nScore: 7'])
    monkeypatch.setattr(mod, 'resolve_backend', lambda **_: backend)
    headers = {'X-Session-ID': 'offline-fineproofs'}
    from openreward.api.sandboxes.secrets import build_secrets_header
    headers['X-Secrets'] = build_secrets_header({'chat_api_key': ('offline-key', ['api.openai.com'])})

    def result(response):
        assert response.status_code == 200, response.text
        pieces = []
        for block in response.text.replace('\r\n', '\n').split('\n\n'):
            fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)
            assert fields.get('event') != 'error', fields
            if fields.get('event') in ('chunk', 'end'):
                pieces.append(fields['data'])
        return json.loads(''.join(pieces))

    with TestClient(Server([mod.FineProofsRL]).app) as client:
        assert client.get('/health').status_code == 200
        assert len(client.post('/FineProofsRL/tasks', json={'split': 'train'}).json()['tasks']) == 1
        assert client.post('/create', headers=headers, json={
            'env_name': 'FineProofsRL', 'task_spec': mod.test_tasks[0]}).status_code == 200
        prompt = client.get('/FineProofsRL/prompt', headers=headers)
        assert 'HIDDEN RUBRIC' not in prompt.text
        tools = client.get('/FineProofsRL/tools', headers=headers)
        assert 'submit_proof' in tools.text
        first_result = result(client.post('/FineProofsRL/call', headers=headers, json={
            'name': 'submit_proof', 'input': {'proof': '1 = 1.'}}))
        assert first_result['ok'], first_result
        first = first_result['output']
        assert first['reward'] == 1 and first['finished']
        assert first['metadata']['grader'] == 'gpt-6-sol'
        assert 'HIDDEN RUBRIC' not in json.dumps(first)
        repeated = result(client.post('/FineProofsRL/call', headers=headers, json={
            'name': 'submit_proof', 'input': {'proof': '1 = 1.'}}))
        assert not repeated['ok'] and repeated['reason'] == 'episode_finished'
        assert backend.calls == 1
        assert client.post('/delete', headers=headers).status_code == 200
        assert backend.closed
