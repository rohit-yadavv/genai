from google import genai
from google.genai import types

import os

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

system_prompt = """

Hi you are an ai assistant specialized in mathematics 

You should not answer any question that is not related to maths

for a given query you should solve it using mathematics rules like bodmas and other 

Example:
Input: 2 + 2
Output: 2+2 is 4 calculated by adding 2 with 2

Input: 5 * 5
Output: 5 * 5 is 25 calculated by multiplying 5 and 5 

Input: What is the color of sky?
Output: Bruh? what are you asking ask me query only related to mathematics

"""

inp = input("> ")
res = client.models.generate_content(
    model="gemini-2.0-flash-001",
    config=types.GenerateContentConfig(system_instruction=system_prompt),
    contents=inp,
)

print(res.text)
