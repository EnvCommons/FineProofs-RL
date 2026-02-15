"""
Sample agent using Anthropic's API to test the FineProofs-RL environment.

This agent:
1. Connects to the FineProofs-RL environment
2. Gets a task (mathematical problem)
3. Uses Claude to generate a proof
4. Submits the proof for grading
"""

import anthropic
from openreward import OpenReward
import asyncio
import json


async def test_function():
    or_client = OpenReward()
    ant_client = anthropic.Anthropic()
    MODEL_NAME = "claude-sonnet-4-5"

    environment = or_client.environments.get(name="fineproofs-rl")
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="anthropic")
    example_task = tasks[0]

    print(f"Testing with task: {example_task['id']}")
    print(f"Problem: {example_task['problem'][:200]}...")
    print()

    async with environment.session(task=example_task) as session:
        prompt = await session.get_prompt()
        messages = [{"role": "user", "content": prompt[0].text}]
        finished = False
        print("Starting proof generation...")
        print()

        while not finished:
            message = ant_client.messages.create(
                model=MODEL_NAME,
                max_tokens=4096,
                tools=tools,
                messages=messages
            )

            print(f"Model response: {message}")
            print()
            messages.append(message)

            if message.stop_reason == "tool_use":
                tool_use = next(block for block in message.content if block.type == "tool_use")
                tool_name = tool_use.name
                tool_input = tool_use.input

                print(f"Tool called: {tool_name}")
                print(f"Arguments: {tool_input}")
                print()

                tool_result = await session.call_tool(tool_name, tool_input)
                reward = tool_result.reward
                finished = tool_result.finished

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result.blocks[0].text
                        }
                    ]
                })

                print("Grading result:")
                print(tool_result.blocks[0].text)
                print()
                print(f"Reward: {reward:.3f}")
                print()

                if tool_result.finished:
                    finished = True
                    print("Episode finished!")
                    break


if __name__ == "__main__":
    asyncio.run(test_function())
