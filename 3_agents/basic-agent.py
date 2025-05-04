from openai import OpenAI
import requests
import os
import json

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def run_command(command):
    result = os.system(command=command)
    return result


def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city",
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns ouput",
    },
}


system_prompt = """
You are a helpful AI assistant that follows a Plan-Action-Observe-Final reasoning process. 
For each user query, proceed as follows:

- Plan: Analyze the user's query and decide if a tool should be used. In this step, output a JSON object with "step": "plan" and a brief explanation of the plan or understanding in "content". 
- Action: If a tool is needed, output a JSON object with "step": "action", the tool name in "function", and the tool input in "input". If no tool is needed, skip this step. 
- Observe: After calling a tool, output a JSON object with "step": "observe" and put the tool's response in "result". If the tool fails or returns an error, you may put an error message in "result" here. 
- Final: Once planning and any tool calls are done, output a JSON object with "step": "final" and your answer to the user in "content". 

Important rules: Always output exactly one JSON object per step (no additional text or formatting). Each JSON must follow this schema and use these keys correctly:

{
  "step": "plan/action/observe/final",
  "content": "string or explanation (for plan and final steps)",
  "function": "tool name (for action step only)",
  "input": "tool input string (for action step only)",
  "result": "tool output or error (for observe step only)"
}

Available Tools:
- get_weather: Takes a city name as an input and returns the current weather for the city
- run_command: Takes a command as input to execute on system and returns ouput


Example:
User Query: What is the weather of new york?
Output: {{"step": "plan", "content": "The user is interseted in weather data of new york" }}
Output: {{"step": "plan", "content": "From the available tools I should call get_weather" }}
Output: {{"step": "action", "function": "get_weather", "input": "New York"}}
Output: {{"step": "observe","result": "12°C"}}
Output: {{"step": "final",  "content": "The current temperature in New York is 12°C."}}


"""


messages = [{"role": "system", "content": system_prompt}]

while True:
    user_query = input("> ")
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gemini-2.5-flash-preview-04-17",
            response_format={"type": "json_object"},
            messages=messages,
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_output)})

        if parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get('content')}")
            continue

        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if available_tools.get(tool_name, False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps({"step": "observe", "result": output}),
                    }
                )
                continue

        if parsed_output.get("step") == "final":
            print(f"🤖: {parsed_output.get('content')}")
            break
