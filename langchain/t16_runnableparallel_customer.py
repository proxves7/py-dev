import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

sentiment_prompt = ChatPromptTemplate.from_template("分析以下文本的情感傾向，你的回答必須使用繁體中文並且使用台灣慣用語: {text}")
topic_prompt = ChatPromptTemplate.from_template("提取以下文本的主要主題，你的回答必須使用繁體中文並且使用台灣慣用語: {text}")
summary_prompt = ChatPromptTemplate.from_template("為以下文本生成一個簡短的摘要，你的回答必須使用繁體中文並且使用台灣慣用語: {text}")

sentiment_chain = sentiment_prompt | model | StrOutputParser()
topic_chain = topic_prompt | model | StrOutputParser()
summary_chain = summary_prompt | model | StrOutputParser()

text_analyzer = RunnableParallel(
    sentiment=sentiment_chain,
    topic=topic_chain,
    summary=summary_chain
)

text = "上個星期入住了這家飯店，整體感覺還不錯，服務人員態度很好，房間也很乾淨，下次還會再來。"
results = text_analyzer.invoke({"text": text})

print("情感分析:", results["sentiment"])
print("主題:", results["topic"])
print("摘要:", results["summary"])
