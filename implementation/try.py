import os
from openai import OpenAI
client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

completion = client.completions.create(
  model="gpt-4o-mini",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)

print(completion.choices[0].text)