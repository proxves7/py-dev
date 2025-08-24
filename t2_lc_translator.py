import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

parser = StrOutputParser()
chain = model | parser

user_input = "知之為知之，不知為不知，是知也。"
message = [
    SystemMessage(content="將以下的內容翻譯為英文。"),
    HumanMessage(content=user_input),
]

print(chain.invoke(message))
