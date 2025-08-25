import os
import google.generativeai as genai
from dotenv import dotenv_values

config = dotenv_values(".env")
genai.configure(api_key=config.get("GOOGLE_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
)

user_input = "如何獲得幸福人生？"
response = model.generate_content(
    user_input,
)

print("Q: " + user_input)
print("A: " + response.text)
