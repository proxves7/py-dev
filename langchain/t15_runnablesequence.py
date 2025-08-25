import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

chinese_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一位短文寫作高手，將以使用者指定的主題進行寫作創作"), ("user", "{topic}")]
)

translation_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一位中英文語言專家，負責中文英的翻譯工作，翻譯的品質必須確保不可以失去文章內容原意，你的輸出結果必須符合以下格式\n\n 中文文章:..... ; 英文文章:...."), ("user", "{chinese_article}")]
)

work_flow = RunnableSequence(
    chinese_prompt, model, translation_prompt, model, StrOutputParser()
)

# 使用 LCEL 表達式建立工作流程
# work_flow = chinese_prompt | llm | translation_prompt | llm | StrOutputParser()

print(work_flow.invoke({"topic":"生成式AI的未來"}))
