# FineProofs-RL

OpenReward environment for mathematical proof generation and evaluation using LLM-based grading.

## Overview

FineProofs-RL is a single-turn evaluation environment where agents are presented with olympiad-level mathematical problems and must generate proofs that are graded against detailed rubrics. The environment uses **gpt-5-mini** for automated grading with a 0-7 point scale, allowing for partial credit.

**Key Features:**
- 5,227 olympiad-level mathematical problems
- LLM-based grading with detailed rubrics
- 0-7 score scale with partial credit
- Simple reward normalization (score/7)
- Multi-provider support (OpenAI, Anthropic, Google Gemini)

## Dataset

**Source**: [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

The dataset contains mathematical problems from two sources:
- **AoPS** (Art of Problem Solving): 3,794 problems (72.7%)
- **Olympiads**: 1,433 problems (27.3%)

Each problem includes:
- Problem statement (shown to agent)
- Detailed grading rubric (hidden from agent, used for evaluation)
- Source identifier

## Task Format

Agents receive:
```
Problem: [Mathematical problem statement]
```

Agents must:
1. Analyze the problem
2. Generate a mathematical proof
3. Submit the proof using the `submit_proof` tool

## Grading System

Proofs are evaluated by **gpt-5-mini with medium reasoning effort** using the following process:

1. **Rubric-based evaluation**: Each problem has a detailed rubric with checkpoints (e.g., "Key Lemma (3 pts)", "Construction (2 pts)")
2. **Score assignment**: Integer score from 0-7 with partial credit
3. **Reward computation**: Linear normalization `reward = score / 7.0`
4. **Feedback**: Detailed grading explanation (2-3 sentences)

**Note**: Rubrics are **NOT** shown to agents - they must infer proof quality criteria.

## Installation

### Local Development

```bash
# Clone or navigate to this directory
cd FineProofs-RL

# Download dataset
curl -L -o fineproofs_train.parquet \
https://huggingface.co/datasets/lm-provers/FineProofs-RL/resolve/main/data/train-00000-of-00001.parquet

# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
```

### Docker

```bash
# Build image
docker build -t fineproofs-rl:latest .

# Run container
docker run -p 8000:8000 fineproofs-rl:latest
```

## Usage

### OpenAI Test Agent

```bash
export OPENAI_API_KEY=your_key_here
python sample_agent.py
```

### Anthropic Test Agent

```bash
export ANTHROPIC_API_KEY=your_key_here
python sample_ant_agent.py
```

### Google Gemini Test Agent

```bash
export GOOGLE_API_KEY=your_key_here
python sample_gemini_agent.py
```

### Programmatic Usage

```python
from openreward import OpenReward

or_client = OpenReward()
environment = or_client.environments.get(name="EnvCommons/FineProofs-RL")

# List available tasks
tasks = await environment.list_tasks(split="train")

# Create a session (requires OpenAI API key for grading)
async with environment.session(
    task=tasks[0],
    secrets={"openai_api_key": "your-key"}
) as session:
    # Get prompt
    prompt = await session.get_prompt()

    # Submit proof
    result = await session.call_tool("submit_proof", {
        "proof": "Your mathematical proof here..."
    })

    print(f"Score: {result.metadata['score']}/7")
    print(f"Reward: {result.reward:.2f}")
    print(f"Feedback: {result.blocks[0].text}")
```

## Tools

### `submit_proof`

Submit your proof for grading.

**Parameters**:
- `proof` (str): Your mathematical proof

**Returns**:
- `blocks`: Grading feedback and score
- `metadata`:
  - `task_id`: Task identifier
  - `score`: Integer score (0-7)
  - `reward`: Normalized reward (0.0-1.0)
  - `grading_response`: Detailed feedback
- `reward`: Float reward value (0.0-1.0)
- `finished`: Always `True` (single-turn)

## Example

```python
# Agent receives this prompt:
"""
Problem: Let $ABC$ be a triangle with $AB = 13$, $BC = 14$, and $CA = 15$.
Let $D$ be the foot of the altitude from $A$ to $BC$. Find $BD$.
"""

# Agent generates proof and submits:
await session.call_tool("submit_proof", {
    "proof": """
    We can use the Pythagorean theorem on triangle ABD.
    Let BD = x, then CD = 14 - x.

    By Pythagorean theorem on ABD: AD^2 = 13^2 - x^2
    By Pythagorean theorem on ACD: AD^2 = 15^2 - (14-x)^2

    Setting equal: 13^2 - x^2 = 15^2 - (14-x)^2
    169 - x^2 = 225 - (196 - 28x + x^2)
    169 - x^2 = 225 - 196 + 28x - x^2
    169 = 29 + 28x
    140 = 28x
    x = 5

    Therefore, BD = 5.
    """
})

# Receives grading:
# Analysis: The proof correctly applies the Pythagorean theorem to both right
# triangles and solves algebraically for BD. All steps are justified and the
# final answer is correct.
# Score: 7
#
# Score: 7/7 | Reward: 1.00
```

## Environment Details

**Type**: Single-turn evaluation (no sandbox)
**Base class**: `Environment` (from `openreward.environments`)
**Namespace**: `EnvCommons/FineProofs-RL`
**Splits**: `["train"]` (5,227 tasks)

## Development

### Testing

```bash
# Syntax check
python -m py_compile server.py sample_agent.py

# Run server
python server.py

# Test with agent (in another terminal)
export OPENAI_API_KEY=your_key_here
python sample_agent.py
```

### Docker Testing

```bash
# Build
docker build -t fineproofs-rl:test .

# Run
docker run -p 8000:8000 fineproofs-rl:test

# Test from host
python sample_agent.py
```

## Deployment

This environment is deployed at `EnvCommons/FineProofs-RL` on OpenReward.

To deploy your own version:

1. Fork or clone this repository
2. Push to GitHub
3. Go to https://openreward.ai/environments/new
4. Connect your GitHub repository
5. Configure build settings
6. Deploy

## Technical Details

### Architecture

- **Pattern**: AIME2025 single-turn
- **Grading model**: gpt-5-mini with medium reasoning effort (via responses API)
- **Data loading**: Module-level pandas load from parquet
- **Rubric storage**: Module-level dict (not exposed to agent)
- **Reward formula**: `score / 7.0` (simple linear normalization)

### API Key Requirements

The environment requires an OpenAI API key for grading. This is passed via the `secrets` parameter:

```python
async with environment.session(
    task=task,
    secrets={"openai_api_key": "your-key"}
) as session:
    ...
```

**Important**: The environment does NOT fall back to environment variables. The API key must be explicitly provided.

### Score Parsing

The grading system uses robust score parsing:
1. Primary: Regex match for `"Score: X"` pattern
2. Fallback: Extract last number in response
3. Default: 0 if parsing fails
4. Clamping: All scores clamped to [0, 7] range

### Error Handling

- **Missing API key**: Raises `ValueError` on initialization
- **Grading API failure**: Returns score=0, reward=0.0 with error message
- **Score parsing failure**: Multiple fallback strategies with default=0
- **Invalid task ID**: Pydantic validation catches during initialization

## References

- **Dataset**: https://huggingface.co/datasets/lm-provers/FineProofs-RL
- **OpenReward**: https://openreward.ai
- **GitHub**: https://github.com/EnvCommons/FineProofs-RL

## License

This environment follows the OpenReward licensing terms. The dataset is provided by lm-provers and subject to its own license terms.

## Citation

If you use this environment, please cite:

```bibtex
@misc{fineproofs-rl-2024,
  title={FineProofs-RL: Mathematical Proof Generation with Reinforcement Learning},
  author={lm-provers},
  year={2024},
  url={https://huggingface.co/datasets/lm-provers/FineProofs-RL}
}
```
