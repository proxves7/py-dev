import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

describe_prompt = PromptTemplate(
    input_variables=["city"],
    template="請用一段優雅的文字描述這個城市：### {city} ###",
)

translate_prompt = PromptTemplate(
    input_variables=["description"],
    template="請將以下描述翻譯成英文：### {description} ###"
)

describe_chain = LLMChain(llm=model, prompt=describe_prompt)
translate_chain = LLMChain(llm=model, prompt=translate_prompt)

chain = SimpleSequentialChain(chains=[describe_chain, translate_chain])

result = chain.invoke("高雄")
print(result["output"])
