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
        return output or "Command runned Successfully, \n No output."
    except Exception as e:
        return f"Error: {str(e)}"


# ----- Tool Router -----
def tool_router(state: State) -> State:
    tool_prompt = """
    You are a tool router.
    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput

    Identify the tool to use and the argument to pass.

    Respond in JSON like:
    {
        "tool": "get_weather",
        "arg": "Delhi"
    }

    or

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


# ----- LLM Decision Logic -----
def decide_path(
    state: State,
) -> Literal["tool_router", "solve_coding_question", "solve_simple_question"]:
    decision_prompt = """
    You are an AI decision engine. Based on the user's query, choose the correct handling path from below.

    Your options:
    1. "tool_router" — If the user is asking to run ANY command-line command (e.g., git, npm, ls, curl, mkdir, python, etc.) or requesting weather information.
    2. "solve_coding_question" — If the user is asking a coding-related question (e.g., how to fix an error, write a function, explain code, debug something).
    3. "solve_simple_question" — If the question is general (e.g., definitions, explanations, or anything not needing a tool or code).

    Respond with ONLY ONE of the following strings: 
    "tool_router", "solve_coding_question", or "solve_simple_question".

    Do not include any explanations or extra text — just one of the above three words.
    """

    messages = [
        {"role": "system", "content": decision_prompt},
        {"role": "user", "content": state["user_query"]},
    ]

    res = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17", messages=messages
    )
    decision = res.choices[0].message.content.strip().lower()

    if "tool_router" in decision:
        return "tool_router"
    elif "coding" in decision or "solve_coding_question" in decision:
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

# Edges
graph_builder.add_conditional_edges(START, decide_path)
graph_builder.add_edge("tool_router", END)
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
