# FineProofs-RL

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/FineProofsRL)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

## Description

FineProofs-RL is an environment for evaluating mathematical proof generation. Given an Olympiad-level math problem, the agent must write a complete proof. An LLM grader scores the proof on a 0-7 scale using a detailed rubric.

## Capabilities

- Writing rigorous mathematical proofs
- Olympiad-level problem solving
- Formal mathematical reasoning
- Proof structure and argumentation

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Tasks

There is one split in this environment:

- **train**: 5,227 problems sourced from:
  - **Art of Problem Solving (AoPS)**: 3,794 problems
  - **International Olympiad competitions** (IMO, APMO, USAMO, USAJMO): 1,433 problems

Each task presents a mathematical problem statement to the agent. The grading rubric is hidden from the agent and used only by the LLM grader during evaluation.

## Reward Structure

This is a single-turn, LLM-graded reward environment. The agent submits a proof via the `submit_proof` tool. An LLM grader (gpt-5-mini) evaluates the proof against a hidden rubric on a 0-7 integer scale with partial credit. The reward is normalized as `score / 7.0`, giving a continuous range from 0.0 to 1.0.

## Data

The dataset is sourced from [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL) on Hugging Face. It contains `fineproofs_train.parquet` with mathematical problems, grading rubrics, and reward annotations. Data files are stored on the OpenReward platform.

## Tools

Agents have access to a single tool:

- **`submit_proof`**: Submit a mathematical proof for rubric-based grading. Accepts a `proof` string parameter. Returns the grader score (0-7), normalized reward (0.0-1.0), and grader feedback. This tool ends the episode.

## Time Horizon

Single-turn. The agent reads the problem and submits one proof.

## Environment Difficulty

Tasks are Olympiad-level math problems from IMO, APMO, USAMO, USAJMO, and AoPS competitions. The rubric-based grading requires proofs that demonstrate both correct reasoning and clear mathematical argumentation.

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
