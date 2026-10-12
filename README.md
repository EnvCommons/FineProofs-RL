# FineProofs-RL

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/FineProofs-RL) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

## Description

FineProofs-RL is an environment for evaluating mathematical proof generation. Given an Olympiad-level math problem, the agent must write a complete proof. An LLM grader scores the proof on a 0-7 scale using a detailed rubric.

## Capabilities

- Writing rigorous mathematical proofs
- Olympiad-level problem solving
- Formal mathematical reasoning
- Proof structure and argumentation

## Compute Requirements

This is a single-turn environment with no sandbox.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Tasks

There is one split in this environment:

- **train**: 5,227 problems sourced from:
  - **Art of Problem Solving (AoPS)**: 3,794 problems
  - **International Olympiad competitions** (IMO, APMO, USAMO, USAJMO): 1,433 problems

Each task presents a mathematical problem statement. The grader uses checkpoint-based rubrics with specific point allocations for major conceptual steps, partial credit for intermediate progress, and deductions for logical gaps vs. arithmetic errors.

## Reward Structure

This is a single-turn environment with continuous reward (0.0–1.0):

The agent submits a proof via the `submit_proof` tool. An LLM grader (`gpt-6-sol` with medium reasoning effort) grades the proof at IMO standard on a 0–7 integer scale, using a hidden rubric as a guide to partial credit for one valid approach. A complete, rigorous proof earns 7 even if it uses a different method from the rubric. The grader is instructed to be strict: steps asserted without justification ("one can check", "verified by a computer algebra system") count as gaps, and false claims, circular reasoning, missing cases and wrong answers are penalised; length, confidence and formatting are not rewarded.

- **7/7 (reward 1.0)**: Complete and rigorous proof, by any method
- **6/7 (reward 0.86)**: Complete proof with a minor slip
- **1-5/7 (reward 0.14–0.71)**: Partial progress, credited using the rubric
- **0/7 (reward 0.0)**: No substantial progress, or a fatal flaw

The final reward is `score / 7.0`, normalized to the 0.0–1.0 range.

## Data

Data consists of a single Parquet file (`fineproofs_train.parquet`) containing 5,227 olympiad-level mathematical problems. Each instance includes the problem statement, source attribution (AoPS or competition name), and a detailed grading rubric with checkpoints for evaluation.

Source: [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

## Tools

Agents have access to a single tool:

- **`submit_proof`**: Submit a mathematical proof for rubric-based grading. Accepts a `proof` string parameter. Returns the grader score (0-7) and normalized reward (0.0-1.0). This tool ends the episode.

## Time Horizon

FineProofs-RL is a single-turn environment. The agent reads the problem and submits one proof for a total of one tool call.

## Environment Difficulty

The dataset includes 128 rollouts per problem from Qwen3-4B-Thinking, providing difficulty estimates:

| Difficulty | reward_mean | Proportion |
|------------|-------------|------------|
| Easier | < 0.3 | ~30% |
| Medium | 0.3 – 0.7 | ~45% |
| Challenging | ≥ 0.7 | ~25% |

Overall dataset mean reward is 0.41. Problems span IMO, APMO, USAMO, USAJMO, and AoPS competitions.

## Other Environment Requirements

A grader API key is required in session secrets as `openai_api_key` or `chat_api_key`. Grading uses OpenReward's `resolve_backend` with the OpenAI provider. With `openai_api_key`, the backend follows `OPENAI_BASE_URL`; `chat_api_key`, `CHAT_API_KEY` or `CHAT_BASE_URL` select a dedicated judge channel. The configured endpoint must serve `gpt-6-sol` and accept `reasoning_effort="medium"` in Chat Completions. When passing `chat_api_key` through the OpenReward SDK, specify its allowed domain: `secrets={"chat_api_key": (key, ["api.openai.com"])}` (or the host of your judge gateway).

Requests use standard service tier, no temperature parameter and a 24,000-token completion limit (including reasoning). The backend uses the SDK's default transient-error retry policy. Incomplete or scoreless replies are retried up to three grading attempts; exhausted retries remain grading errors and are not converted into zero rewards. Grader analysis remains private.

## Safety

Agents in FineProofs-RL write mathematical proofs in a standard environment. The environment does not present direct safety risks.

## Citations

```bibtex
@misc{qednano2026,
  title = {QED-Nano: Teaching a Tiny Model to Prove Hard Theorems},
  author = {Yuxiao Qu and Amrith Setlur and Jasper Dekoninck and Edward Beeching and Jia Li and Ian Wu and Lewis Tunstall and Aviral Kumar},
  year = {2026},
  howpublished = {https://huggingface.co/spaces/lm-provers/qed-nano-blogpost},
  note = {Blog post}
}
```


## Local Validation

Run `python -m pytest tests.py -q` with the dependencies from `requirements.txt`. Tests use a temporary one-task dataset and scripted model responses, without paid calls. The real dataset can be supplied as `fineproofs_train.parquet` in the working directory or through `/orwd_data` on the platform. See [VALIDATION.md](VALIDATION.md) for the local server and live model checks.
