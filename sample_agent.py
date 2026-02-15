"""
Sample agent using OpenAI's Responses API to test the FineProofs-RL environment.

This agent:
1. Connects to the FineProofs-RL environment
2. Gets a task (mathematical problem)
3. Uses GPT to generate a proof
4. Submits the proof for grading
"""

from openai import OpenAI
from openreward import AsyncOpenReward
import asyncio
import os
import json


async def test_function():
    or_client = AsyncOpenReward()
    oai_client = OpenAI()
    MODEL_NAME = "gpt-4o"

    environment = or_client.environments.get(name="fineproofs-rl", base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")
    example_task = tasks[0]

    #print(f"Testing with task: {example_task['id']}")
    #print(f"Problem: {example_task['problem'][:200]}...")
    print()

    async with environment.session(task=example_task, secrets={"openai_api_key": os.getenv("OPENAI_API_KEY")}) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False
        print("Starting proof generation...")
        print()

        while not finished:
            response = oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list
            )

            print(f"Model output: {response.output}")
            print()

            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    print(f"Tool called: {item.name}")
                    print(f"Arguments: {json.loads(str(item.arguments))}")
                    print()

                    tool_result = await session.call_tool(item.name, json.loads(str(item.arguments)))
                    reward = tool_result.reward
                    finished = tool_result.finished

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({
                            "result": tool_result.blocks[0].text
                        })
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
