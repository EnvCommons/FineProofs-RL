# FineProofs-RL

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/FineProofsRL) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

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

Each task presents a mathematical problem statement to the agent. The grading rubric is hidden from the agent and used only by the LLM grader during evaluation.

## Reward Structure

This is a single-turn environment with continuous reward (0.0–1.0):

The agent submits a proof via the `submit_proof` tool. An LLM grader (gpt-5-mini with medium reasoning effort) evaluates the proof against a hidden rubric on a 0–7 integer scale:

- **7/7 (reward 1.0)**: Complete, rigorous proof with all checkpoints satisfied
- **4-6/7 (reward 0.57–0.86)**: Partial credit for correct reasoning with minor gaps
- **1-3/7 (reward 0.14–0.43)**: Some correct steps but significant issues
- **0/7 (reward 0.0)**: Incorrect or missing proof

The final reward is `score / 7.0`, normalized to the 0.0–1.0 range.

## Data

Data consists of a single Parquet file (`fineproofs_train.parquet`) containing 5,227 olympiad-level mathematical problems. Each instance includes the problem statement, source attribution (AoPS or competition name), and a detailed grading rubric with checkpoints for evaluation.

Source: [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

## Tools

Agents have access to a single tool:

- **`submit_proof`**: Submit a mathematical proof for rubric-based grading. Accepts a `proof` string parameter. Returns the grader score (0-7), normalized reward (0.0-1.0), and grader feedback. This tool ends the episode.

## Time Horizon

Single-turn. The agent reads the problem and submits one proof.

## Environment Difficulty

Tasks are Olympiad-level math problems from IMO, APMO, USAMO, USAJMO, and AoPS competitions. Recent evaluations on similar benchmarks show that LLMs struggle significantly with proof generation:

| Model | 2025 USAMO Score |
|-------|------------------|
| Gemini-2.5-Pro | 25% |
| All other models | <5% |

The QED-Nano project demonstrates that specialized 4B-parameter models trained with reinforcement learning can approach larger models' performance on proof tasks, but the rubric-based grading requires both correct reasoning and clear mathematical argumentation.

## Other Environment Requirements

OpenAI API key required for LLM-based proof grading. Pass via `secrets={"openai_api_key": "..."}`.

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
