import os
import json
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableWithFallbacks
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
# model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


advanced_prompt = ChatPromptTemplate.from_template("請回答以下問題：{question}")
base_prompt = ChatPromptTemplate.from_template("請回答以下問題：{question}")

advanced_llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
base_llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

advanced_chain = RunnableSequence(
    advanced_prompt,
    advanced_llm
)

base_chain = RunnableSequence(
    base_prompt,
    base_llm
)

def unstable_advanced_model(query):
    if time.time() % 2 == 0:
        raise Exception("LLM Service unavailable")
    return advanced_chain.invoke(query)

def predefined_fallback(query):
    return "很抱歉，目前無法回應您的問題，請洽客服專線。"

qa_chain = RunnableLambda(unstable_advanced_model)

qa_system = RunnableWithFallbacks(
    runnable=qa_chain,
    fallbacks=[base_chain, RunnableLambda(predefined_fallback)]
)

for _ in range(5):
    try:
        result = qa_system.invoke({"question": "什麼是生成式AI"})
        print(f"回答: {result.content}")
    except Exception as e:
        print(f"錯誤: {str(e)}")
    print()
    time.sleep(1)
