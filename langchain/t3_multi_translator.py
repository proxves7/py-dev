import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "將以下的內容翻譯為{language}"),
     ("user", "{text}")]
)
chain = prompt_template | model | parser

target_language = "日文"
user_input = "知之為知之，不知為不知，是知也。"

print(chain.invoke({"language": target_language, "text": user_input}))
