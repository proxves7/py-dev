import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


language_identification_prompt= ChatPromptTemplate.from_template(
"Please identify the language of the following text. "
"Respond with 'Chinese' for Chinese, 'English' for English, or 'Other' for any other language. "
"Text: {text}")
language_identification_chain = language_identification_prompt | model | StrOutputParser()

chinese_prompt = ChatPromptTemplate.from_template("你是一位中文客服機器人，請根據用戶的問題提供中文回應。###{text}")
chinese_chain = chinese_prompt | model | StrOutputParser()

english_prompt = ChatPromptTemplate.from_template("You are an English customer service bot. Please respond to the user's query in English.###{text}")
english_chain = english_prompt | model | StrOutputParser()


workflow = RunnableSequence(
    {"language": language_identification_chain,
      "text": lambda x: x["text"]},
    RunnableBranch(
        (lambda x: x["language"].strip().lower() == "chinese"
         , chinese_chain),
        (lambda x: x["language"].strip().lower() == "english"
         , english_chain),
        english_chain
    )
)

# text = "上個星期入住了這家飯店，整體感覺還不錯，服務人員態度很好，房間也很乾淨，下次還會再來。"
# text = "I stayed in this hotel last week. The overall feeling is pretty good. The service staff is very friendly and the room is very clean. I will come back next time."
text = "先週このホテルに泊まりました。全体的な雰囲気はとても良く、スタッフはとてもフレンドリーで、次回もまた来ます。"
results = workflow.invoke({"text": text})

print(results)
