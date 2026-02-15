"""
Sample agent using Google Gemini's API to test the FineProofs-RL environment.

This agent:
1. Connects to the FineProofs-RL environment
2. Gets a task (mathematical problem)
3. Uses Gemini to generate a proof
4. Submits the proof for grading
"""

from google import genai
from google.genai import types
from openreward import OpenReward
import asyncio
import json


async def test_function():
    or_client = OpenReward()
    gem_client = genai.Client()
    MODEL_NAME = "gemini-2.5-flash"

    environment = or_client.environments.get(name="fineproofs-rl")
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="google")

    genai_tools = [types.Tool(function_declarations=[f]) for f in tools]
    genai_config = types.GenerateContentConfig(tools=genai_tools)

    example_task = tasks[0]

    print(f"Testing with task: {example_task['id']}")
    print(f"Problem: {example_task['problem'][:200]}...")
    print()

    async with environment.session(task=example_task) as session:
        prompt = await session.get_prompt()
        contents = [
            types.Content(
                role="user", parts=[types.Part(text=prompt[0].text)]
            )
        ]
        finished = False
        print("Starting proof generation...")
        print()

        while not finished:

            response = gem_client.models.generate_content(
                model=MODEL_NAME,
                config=genai_config,
                contents=contents
            )

            print(f"Model response: {response.candidates[0].content}")
            print()

            contents.append(response.candidates[0].content)

            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_call = part.function_call

                    print(f"Tool called: {tool_call.name}")
                    print(f"Arguments: {tool_call.args}")
                    print()

                    tool_result = await session.call_tool(tool_call.name, tool_call.args)

                    reward = tool_result.reward
                    finished = tool_result.finished

                    function_response_part = types.Part.from_function_response(
                        name=tool_call.name,
                        response={"result": json.dumps({
                            "result": tool_result.blocks[0].text
                        })},
                    )

                    contents.append(types.Content(role="user", parts=[function_response_part]))

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
