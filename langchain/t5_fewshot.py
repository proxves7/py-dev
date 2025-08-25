import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from dotenv import dotenv_values

from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
parser = StrOutputParser()

example_prompt = HumanMessagePromptTemplate.from_template("{description}") + AIMessagePromptTemplate.from_template("{classification}")

examples = [
    {
        "description": "食物偏甜",
        "classification": "南部人",
    },
    {
        "description": "食物偏鹹",
        "classification": "北部人",
    },
    {
        "description": "滷肉飯",
        "classification": "北部人",
    },
    {
        "description": "肉燥飯",
        "classification": "南部人",
    },
    {
        "description": "搭乘大眾運輸，不怕走路",
        "classification": "北部人",
    },
    {
        "description": "騎摩托車，不待轉",
        "classification": "南部人",
    },
    {
        "description": "講話婉轉，不直接",
        "classification": "北部人",
    },
    {
        "description": "講話直接",
        "classification": "南部人",
    },
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [("system", "請根據以下描述，判斷是哪一種人："),
     few_shot_prompt,
     ("human", "{input}"),]
)

chain = final_prompt | model | parser

user_input = "醬油喜歡有甜味"
response = chain.invoke({"input": user_input})
print("描述：", user_input)
print("分類：", response)

print("==========")
user_input = "熱情大方，講話直接"
response = chain.invoke({"input": user_input})
print("描述：", user_input)
print("分類：", response)
