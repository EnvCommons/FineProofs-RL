# GPT-6 Sol grader validation

Tested locally on 2026-10-12, before pushing the grader change to `main`.

## Configuration

- OpenReward 0.1.160, OpenAI Python SDK 2.24.0.
- Real `fineproofs_train.parquet`: 5,227 tasks in the `train` split.
- Grader: `gpt-6-sol`, `reasoning_effort="medium"`, standard service tier,
  `max_completion_tokens=24000`, no temperature parameter.
- Grading uses `openreward.chat_backends.resolve_backend(provider="openai", secrets=secrets)`.

## Offline checks

```sh
python -m py_compile server.py tests.py
python -m pytest tests.py -q
```

Result: **11 tests passed**. Coverage includes exact model request options,
gateway and dedicated judge routing, required session credentials, scoreless and
incomplete reply retries, provider errors without fabricated zero rewards, rubric
privacy, repeated submissions and backend cleanup. A FastAPI HTTP test also
creates a session, reads its prompt and tools, grades a proof, verifies that the
framework rejects further calls after the episode ends, and deletes the session.

These tests use a temporary one-task dataset and scripted judge responses; they
make no paid API calls. One unrelated Google GenAI/Pydantic deprecation warning
was emitted by the installed dependencies.

## Live local-server check

The real OpenReward async client connected to a local Uvicorn server running
`Server([FineProofsRL])`. The server loaded all 5,227 real tasks. Task `4` asks for

\[
\int_0^{\pi/2}\frac{dx}{4-3\cos^2 x}.
\]

The submitted solution uses `t = tan(x)`, transforms the integral to
`integral_0^infinity dt / (1 + 4t^2)`, and evaluates it as `pi/4`, including the
endpoint limit justification. The actual judge response completed with
`finish_reason="stop"`; the SDK received **score 7/7, reward 1.0, finished=true**,
with `grader="gpt-6-sol"` and `grader_reasoning_effort="medium"` in metadata.
The public prompt contained only the problem statement; the result did not
expose the rubric or grader analysis.

The test allowed two provider requests, disabled hidden SDK retries for that
test, and reserved spending against the existing authorized budget before each
request. The first request completed grading but its test report writer failed
on an SDK dataclass serialization mismatch. The corrected writer recorded the
second request end to end; both calls' usage is retained in the artifact.

| Call | Input tokens | Output tokens | Estimated API cost |
| --- | ---: | ---: | ---: |
| Initial call, report writer failed | 784 | 94 | $0.002508 |
| Completed end-to-end check | 784 | 89 | $0.002458 |
| Total | 1,568 | 183 | **$0.004966** |

Neither call reported cached tokens, cache writes or reasoning tokens. The
request still explicitly used medium reasoning. Costs use the published
[GPT-6 Sol standard rates](https://developers.openai.com/api/docs/models/gpt-6-sol)
of $2 per million uncached input tokens and $10 per million output tokens.
At the completed call's usage, 16 equivalent grades would cost approximately
**$0.0393** before retries. This is a small integration example; harder problems,
longer submissions and retries can cost more. It is not a dataset-wide grading
accuracy evaluation.

The [machine-readable result](validation/gpt-6-sol-local.json) includes the exact
candidate, returned reward, token usage, versions, and server/dataset hashes.
The existing local dataset was reused; no Docker image was built locally.
