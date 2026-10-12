"""
FineProofs-RL Environment

Mathematical proof generation and evaluation environment using LLM-based grading.
Agents are presented with olympiad-level problems and must generate proofs that are
graded against detailed rubrics (0-7 scale) by gpt-6-sol with medium reasoning effort.

Dataset: HuggingFace lm-provers/FineProofs-RL (5,227 problems)
"""

from __future__ import annotations

import re
import pandas as pd
from pydantic import BaseModel
import os

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool
from openreward.chat_backends import resolve_backend

JUDGE_MODEL = "gpt-6-sol"
JUDGE_REASONING_EFFORT = "medium"
JUDGE_MAX_COMPLETION_TOKENS = 24_000

# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1

# Grader replies without a score are re-sampled this many times before the call fails.
GRADER_SCORE_ATTEMPTS = 3



# Grader template for 0-7 scoring with full rubric
GRADER_TEMPLATE = """You are an expert olympiad grader. Grade the submitted proof at IMO standard.

**Problem:**
{problem}

**Submitted Proof:**
{proof}

**Grading Rubric (a guide to partial credit for one valid approach; 0-7 points):**
{rubric}

**Rules:**
1. A complete and rigorous proof earns 7 even if it uses a different method from the rubric. Use the rubric to award partial credit to incomplete solutions.
2. Be strict. A step asserted without justification ("routine computation", "one can check", "verified by a computer algebra system") is a gap unless it is genuinely trivial. Penalise false claims, circular reasoning, assuming the conclusion, missing cases and wrong answers.
3. Do not reward length, confidence or formatting.
4. 7 = complete and rigorous; 6 = complete with a minor slip; 1-5 = partial progress; 0 = no substantial progress or a fatal flaw.

**Output Format:**
Analysis: [2-4 sentences]
Score: [Integer from 0 to 7]
"""


# Data loading (module level) - loads parquet and extracts task info
if os.path.exists("/orwd_data"):
    test_tasks = pd.read_parquet("/orwd_data/fineproofs_train.parquet").to_dict(orient="records")
else:
    test_tasks = pd.read_parquet("fineproofs_train.parquet").to_dict(orient="records")

# Extract rubrics into separate dict (not exposed to agent)
RUBRICS_DICT = {}
for i, task in enumerate(test_tasks):
    task_id = str(i)
    RUBRICS_DICT[task_id] = task.get("rubrics", "")

    # Clean task spec to only include: id, problem, source
    keep_keys = {"problem", "source"}
    keys_to_remove = set(task.keys()) - keep_keys
    for k in keys_to_remove:
        del task[k]

    # Add task ID
    task["id"] = task_id


# Pydantic models for validation
class TaskSpec(BaseModel):
    """Task specification for FineProofs-RL"""
    id: str
    problem: str
    source: str


class ProofInput(BaseModel):
    """Input schema for submit_proof tool"""
    proof: str


class FineProofsRL(Environment):
    """
    Environment for mathematical proof generation and evaluation.

    Agents are presented with olympiad-level math problems and must submit proofs.
    Proofs are graded by gpt-6-sol with medium reasoning effort using detailed rubrics (0-7 scale).
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        # Require a session credential. The framework backend preserves gateway
        # routing and supports a dedicated judge endpoint via CHAT_* settings.
        api_key = secrets.get("chat_api_key") or secrets.get("openai_api_key")
        if not api_key or not api_key.strip():
            raise ValueError(
                "OpenAI API key required in secrets for grading. "
                "Pass openai_api_key or chat_api_key."
            )
        self.client = resolve_backend(provider="openai", secrets=secrets)

        # Retrieve rubric for this task (hidden from agent)
        self.rubric = RUBRICS_DICT.get(self.config.id, "")

        # Number of proofs submitted this session — only the first is graded/rewarded
        self.submitted = 0

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits"""
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return list of tasks for the given split"""
        if split != "train":
            return []
        return test_tasks

    def get_prompt(self) -> list[TextBlock]:
        """Return prompt for this task (problem only, NO rubric)"""
        return [TextBlock(type="text", text=self.config.problem)]

    @tool
    async def submit_proof(self, params: ProofInput) -> ToolOutput:
        """
        Submit your proof for grading. This will end the episode.

        The proof will be evaluated against a rubric (0-7 scale) by an expert grader.
        You will receive a score and reward.
        """
        # Only the first submission is graded and rewarded
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(
                    type="text",
                    text="A proof has already been submitted for this task. This episode is over: the proof is not re-graded, and repeat submissions are penalised (reward -0.1).",
                )],
                metadata={
                    "task_id": self.config.id,
                    "already_submitted": True,
                    "submission_count": self.submitted,
                },
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        # Grade the proof
        grading_result = await self._grade_proof(params.proof)

        score = grading_result["score"]
        reward = grading_result["reward"]

        # Format display message. The grader's analysis is not shown: it is
        # written against the hidden rubric and names its checkpoints.
        display_text = f"**Score: {score}/7 | Reward: {reward:.2f}**"

        self.submitted += 1

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "task_id": self.config.id,
                "score": score,
                "reward": reward,
                "grader": JUDGE_MODEL,
                "grader_reasoning_effort": JUDGE_REASONING_EFFORT,
            },
            reward=reward,
            finished=True
        )

    async def _grade_proof(self, proof: str) -> dict:
        """
        Grade proof using gpt-6-sol with medium reasoning effort and rubric-based evaluation.

        Returns dict with:
        - score (int 0-7)
        - reward (float 0.0-1.0)
        - grading_response (str)

        Raises RuntimeError if the grader service fails (e.g. 502) or its reply
        has no score, so the tool call errors instead of finishing the rollout
        with reward 0.
        """
        grader_prompt = GRADER_TEMPLATE.format(
            problem=self.config.problem,
            proof=proof,
            rubric=self.rubric
        )

        score = None
        for attempt in range(GRADER_SCORE_ATTEMPTS):
            try:
                response = await self.client.create(
                    model=JUDGE_MODEL,
                    reasoning_effort=JUDGE_REASONING_EFFORT,
                    max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
                    service_tier="default",
                    messages=[{"role": "user", "content": grader_prompt}]
                )
            except Exception as e:
                # Grader service failure (e.g. 502) must not count as a scored
                # rollout — surface it as a tool error instead.
                raise RuntimeError(f"Grading failed: {e}") from e

            grading_response = response.text

            # Parse score with fallback; a reply without one is re-sampled
            # A truncated/refused reply is not a completed assessment, even if
            # its partial text happens to contain a number or a score label.
            score = self._parse_score(grading_response) if response.finish_reason == "stop" else None
            if score is not None:
                break
            print(f"Grader reply had no score (attempt {attempt + 1}/{GRADER_SCORE_ATTEMPTS})")
        if score is None:
            raise RuntimeError("Grading failed: grader reply had no score")
        reward = score / 7.0  # Simple normalization

        return {
            "score": score,
            "reward": reward,
            "grading_response": grading_response,
        }

    def _parse_score(self, grading_response: str) -> int | None:
        """
        Extract score from grading response with robust fallback.

        Strategy:
        1. Look for "Score: X" pattern (primary)
        2. Fallback: Extract last number in response
        3. Clamp to [0, 7] range

        Returns None if the response has no number at all (e.g. an empty
        reply): that is a grader failure, not a score of 0.
        """
        # Look for "Score: X" pattern
        match = re.search(r"Score:\s*(\d+)", grading_response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(7, score))  # Clamp to [0, 7]

        # Fallback: find last number in response
        numbers = re.findall(r"\b(\d+)\b", grading_response)
        if numbers:
            score = int(numbers[-1])
            return max(0, min(7, score))

        return None

    async def teardown(self) -> None:
        await self.client.aclose()


# Server initialization
if __name__ == "__main__":
    Server([FineProofsRL]).run()
