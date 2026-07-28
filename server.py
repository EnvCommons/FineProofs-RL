"""
FineProofs-RL Environment

Mathematical proof generation and evaluation environment using LLM-based grading.
Agents are presented with olympiad-level problems and must generate proofs that are
graded against detailed rubrics (0-7 scale) by gpt-5-mini with medium reasoning effort.

Dataset: HuggingFace lm-provers/FineProofs-RL (5,227 problems)
"""

from __future__ import annotations

import re
import pandas as pd
import openai
from pydantic import BaseModel
import os

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool


# Grader template for 0-7 scoring with full rubric
GRADER_TEMPLATE = """You are an expert mathematician evaluating olympiad-level proofs.

**Problem:**
{problem}

**Submitted Proof:**
{proof}

**Grading Rubric (0-7 points total):**
{rubric}

**Instructions:**
1. Evaluate the proof against each checkpoint in the rubric
2. Award partial credit for partially correct work
3. Provide brief analysis (2-3 sentences) explaining your scoring
4. Assign an integer score from 0 to 7

**Output Format:**
Analysis: [Your 2-3 sentence explanation]
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
    Proofs are graded by gpt-5-mini with medium reasoning effort using detailed rubrics (0-7 scale).
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        # Get OpenAI client from secrets (NO env var fallback)
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required in secrets for grading. "
                "Pass via secrets={'openai_api_key': 'your-key'}"
            )
        self.client = openai.AsyncClient(api_key=api_key, max_retries=5)

        # Retrieve rubric for this task (hidden from agent)
        self.rubric = RUBRICS_DICT.get(self.config.id, "")

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
        You will receive a score, reward, and detailed feedback.
        """
        # Grade the proof
        grading_result = await self._grade_proof(params.proof)

        score = grading_result["score"]
        reward = grading_result["reward"]
        grading_response = grading_result["grading_response"]

        # Format display message
        display_text = f"{grading_response}\n\n**Score: {score}/7 | Reward: {reward:.2f}**"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "task_id": self.config.id,
                "score": score,
                "reward": reward,
                "grading_response": grading_response,
            },
            reward=reward,
            finished=True
        )

    async def _grade_proof(self, proof: str) -> dict:
        """
        Grade proof using gpt-5-mini with medium reasoning effort and rubric-based evaluation.

        Returns dict with:
        - score (int 0-7)
        - reward (float 0.0-1.0)
        - grading_response (str)

        Raises RuntimeError if the grader service fails (e.g. 502), so the
        tool call errors instead of finishing the rollout with reward 0.
        """
        grader_prompt = GRADER_TEMPLATE.format(
            problem=self.config.problem,
            proof=proof,
            rubric=self.rubric
        )

        try:
            # Use gpt-5-mini with medium reasoning effort
            response = await self.client.chat.completions.create(
                model="gpt-5-mini",
                reasoning_effort="medium",
                messages=[{"role": "user", "content": grader_prompt}]
            )
        except Exception as e:
            # Grader service failure (e.g. 502) must not count as a scored
            # rollout — surface it as a tool error instead.
            raise RuntimeError(f"Grading failed: {e}") from e

        # Extract text from chat completions output
        grading_response = response.choices[0].message.content or ""

        # Parse score with fallback
        score = self._parse_score(grading_response)
        reward = score / 7.0  # Simple normalization

        return {
            "score": score,
            "reward": reward,
            "grading_response": grading_response,
        }

    def _parse_score(self, grading_response: str) -> int:
        """
        Extract score from grading response with robust fallback.

        Strategy:
        1. Look for "Score: X" pattern (primary)
        2. Fallback: Extract last number in response
        3. Default to 0 if parsing fails
        4. Clamp to [0, 7] range
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

        # Default to 0 if parsing fails
        return 0


# Server initialization
if __name__ == "__main__":
    Server([FineProofsRL]).run()
