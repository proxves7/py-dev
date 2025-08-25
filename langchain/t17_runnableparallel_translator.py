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

english_prompt = ChatPromptTemplate.from_template("你是一位英文語言專家，請將以下短文翻譯成英文 : {text}")
japan_prompt = ChatPromptTemplate.from_template("你是一位日文語言專家，請將以下短文翻譯成日文 : {text}")
french_prompt = ChatPromptTemplate.from_template("你是一位法語語言專家，請將以下短文翻譯成法文 : {text}")

english_chain = english_prompt | model | StrOutputParser()
japan_chain = japan_prompt | model | StrOutputParser()
french_chain = french_prompt | model | StrOutputParser()

# 建立 RunnableParallel
text_analyzer = RunnableParallel(
    english=english_chain,
    japan=japan_chain,
    french=french_chain
)

text = "生成式人工智慧是一種人工智慧系統,能夠產生文字、圖像或其他媒體,而提示工程將大大的影響其生成結果。"
results = text_analyzer.invoke({"text": text})

print("英文:", results["english"])
print("日文:", results["japan"])
print("法文:", results["french"])