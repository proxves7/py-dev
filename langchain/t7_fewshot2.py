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


#####
print("==================================================")
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embedding_function,
    Chroma,
    k=1,
)

# find the most similar example
question = "騎機車經過十字路口會直接左轉"
# question = "喜歡醬油有甜味"
selected_examples = example_selector.select_examples({"description": question})

print(question)
print("最相似的例子是 :")
for example in selected_examples:
    for key, value in example.items():
        print(f"{key}: {value}")
    print("\n")

#####
print("==================================================")
few_shot_prompt_v2 = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
)
print(few_shot_prompt_v2.invoke({"description": "喜歡吃甜甜"}))

final_prompt_v2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                    請根據以下精選的參考描述，判斷是北部人還是南部人，
                    只要回答是「北部人」或「南部人」即可，不用解釋：
                    """,
        ),
        few_shot_prompt_v2,
        ("human", "{input}"),
    ]
)

chain_v2 = final_prompt_v2 | model | parser

user_input = "醬油喜歡有甜味"
# user_input = "熱情大方，講話直接"
response = chain_v2.invoke({"input": user_input})
print("描述：", user_input)
print("分類：", response)
