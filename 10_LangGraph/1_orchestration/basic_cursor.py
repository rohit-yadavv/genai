from typing import List, Literal, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from langsmith.wrappers import wrap_openai
from langgraph.graph import StateGraph, START, END

import json
import requests
import os

# Load API keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Wrap LLM
client = wrap_openai(
    OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )
)


# ----- Data Types -----
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class State(TypedDict):
    user_query: str
    messages: List[Message]
    result: str
    steps: List[str]  # To store parsed sub-tasks


# ----- Tools -----
def get_weather(city: str) -> str:
    print(f"🌦 Tool Called: get_weather for {city}")
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}."
        return "Failed to fetch weather info."
    except Exception as e:
        return f"Error: {str(e)}"


def run_command(command: str) -> str:
    print(f"🔧 Tool Called: run_command: {command}")
    try:
        output = os.system(command)
        return output or "Command executed successfully, no output."
    except Exception as e:
        return f"Error: {str(e)}"


# ----- Tool Router -----
def tool_router(state: State) -> State:
    tool_prompt = """
You are a tool router.

Available Tools:
- get_weather: Takes a city name as input and returns the current weather for the city.
- run_command: Takes a command-line command as input to execute on the system and returns the output.

Your task is to identify the appropriate tool to use based on the user's query and extract the necessary argument.

Respond in JSON format like:
{
    "tool": "get_weather",
    "arg": "Delhi"
}

or

{
    "tool": "run_command",
    "arg": "ls"
}

Examples:

User query: "What's the weather in New York?"
Response:
{
    "tool": "get_weather",
    "arg": "New York"
}

User query: "Please list the files in the current directory."
Response:
{
    "tool": "run_command",
    "arg": "ls"
}
"""

    messages = [
        {"role": "system", "content": tool_prompt},
        {"role": "user", "content": state["user_query"]},
    ]

    res = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17",
        response_format={"type": "json_object"},
        messages=messages,
    )

    output = json.loads(res.choices[0].message.content)

    try:
        tool = output.get("tool")
        arg = output.get("arg")

        if tool == "get_weather":
            state["result"] = get_weather(arg)
        elif tool == "run_command":
            state["result"] = run_command(arg)
        else:
            state["result"] = "Tool not recognized."
    except Exception as e:
        state["result"] = f"Tool parsing error: {str(e)}"

    return state


# ----- Simple Answer -----
def solve_simple_question(state: State) -> State:
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + state[
        "messages"
    ]
    res = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17", messages=messages
    )
    state["result"] = res.choices[0].message.content
    return state


# ----- Code Answer -----
def solve_coding_question(state: State) -> State:
    messages = [{"role": "system", "content": "You are a coding assistant."}] + state[
        "messages"
    ]
    res = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17", messages=messages
    )
    state["result"] = res.choices[0].message.content
    return state


# ----- Complex Query Handler -----
def handle_complex_query(state: State) -> State:
    planner_prompt = f"""
You are a task planner for an AI agent. Your job is to break down the user's query into an ordered list of simple, actionable steps.

Each step must be a clear instruction that can either be:
- A command-line operation (e.g., create a file, list directory contents).
- A code generation task (e.g., write a Python class, implement a function).

Respond with a JSON array of strings, where each string is a step.

Examples:

User query: "Create a Python file named 'calculator.py' and write a class to add two numbers."
Response:
[
    "Create a file named 'calculator.py'",
    "Write a Python class in 'calculator.py' that adds two numbers"
]

User query: "Check the current weather in London and then write a Python script that prints it."
Response:
[
    "Get the current weather in London",
    "Write a Python script that prints the weather information for London"
]

User query: "{state['user_query']}"
"""

    messages = [{"role": "user", "content": planner_prompt}]
    res = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17",
        response_format={"type": "json_object"},
        messages=messages,
    )
    steps = json.loads(res.choices[0].message.content.strip())
    state["steps"] = steps

    results = []

    for step in steps:
        sub_state = {
            "user_query": step,
            "messages": [{"role": "user", "content": step}],
            "result": "",
        }

        next_step = decide_path(sub_state)

        if next_step == "tool_router":
            result = tool_router(sub_state)
        elif next_step == "solve_coding_question":
            result = solve_coding_question(sub_state)
        else:
            result = solve_simple_question(sub_state)

        results.append(f"Step: {step}\n{result['result']}")

    state["result"] = "\n\n".join(results)
    return state


# ----- Decision Logic -----
def decide_path(
    state: State,
) -> Literal[
    "tool_router",
    "solve_coding_question",
    "solve_simple_question",
    "handle_complex_query",
]:
    decision_prompt = """
You are an AI decision engine. Based on the user's query, choose the appropriate handling path from the options below.

Options:
1. "tool_router" — Use this if the user is asking to:
   - Run any command-line command (e.g., git, npm, mkdir, ls, curl, python).
   - Check the weather in a specific location.

2. "solve_coding_question" — Use this if the user is:
   - Asking a coding-related question (e.g., how to fix an error, write a function).
   - Requesting code generation or debugging assistance.

3. "solve_simple_question" — Use this for general questions that:
   - Seek definitions, explanations, or factual information.
   - Do not require tool usage or code generation.

4. "handle_complex_query" — Use this if the user's request involves multiple steps, such as:
   - Creating a file and then writing code in it.
   - Performing a sequence of actions that include both command execution and code writing.

Respond with ONLY ONE of the following strings:
"tool_router", "solve_coding_question", "solve_simple_question", "handle_complex_query"

Do not include any explanations or additional text.
"""

    messages = [
        {"role": "system", "content": decision_prompt},
        {"role": "user", "content": state["user_query"]},
    ]

    res = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17", messages=messages
    )
    decision = res.choices[0].message.content.strip().lower()

    if "handle_complex_query" in decision:
        return "handle_complex_query"
    elif "tool_router" in decision:
        return "tool_router"
    elif "solve_coding_question" in decision or "coding" in decision:
        return "solve_coding_question"
    else:
        return "solve_simple_question"


# ----- Graph -----
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("tool_router", tool_router)
graph_builder.add_node("solve_simple_question", solve_simple_question)
graph_builder.add_node("solve_coding_question", solve_coding_question)
graph_builder.add_node("decide_path", decide_path)
graph_builder.add_node("handle_complex_query", handle_complex_query)

# Edges
graph_builder.add_conditional_edges(START, decide_path)
graph_builder.add_edge("tool_router", END)
graph_builder.add_edge("handle_complex_query", END)
graph_builder.add_edge("solve_simple_question", END)
graph_builder.add_edge("solve_coding_question", END)

graph = graph_builder.compile()


# ----- Runner -----
def call_graph():
    print("\n🧠 Ask me anything (weather, commands, coding, etc.)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        state = {
            "user_query": query,
            "messages": [{"role": "user", "content": query}],
            "result": "",
        }

        final = graph.invoke(state)
        print("🤖:", final["result"])


# Run
if __name__ == "__main__":
    call_graph()
