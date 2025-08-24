import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

describe_prompt = PromptTemplate(
    input_variables=["city"],
    template="請描述這個城市：### {city} ###"
)

travel_prompt = PromptTemplate(
    input_variables=["description"],
    template="根據這個描述，為遊客提供一些旅遊指南：### {description} ### "
)

translate_prompt = PromptTemplate(
    input_variables=["travel"],
    template="請將以下內容翻譯成英文：### {travel} ###"
)

describe_chain = LLMChain(llm=model, prompt=describe_prompt, output_key="description")
travel_chain = LLMChain(llm=model, prompt=travel_prompt, output_key="travel")
translate_chain = LLMChain(llm=model, prompt=translate_prompt, output_key="final_advice")

sequential_chain = SequentialChain(
    chains=[describe_chain, travel_chain, translate_chain],
    input_variables=["city"],
    output_variables=["final_advice"]
)

result = sequential_chain.invoke("高雄")
print(result["final_advice"])
