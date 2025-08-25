import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


sentiment_analysis_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="根據這段話分析情緒，並僅回答 'positive' 或 'negative'：'{user_input}'"
)
sentiment_analysis_chain = LLMChain(llm=model, prompt=sentiment_analysis_prompt)

negative_response_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="使用者說了這段話：'{user_input}'。請給出一段安撫的回應。"
)
negative_response_chain = LLMChain(llm=model, prompt=negative_response_prompt)

positive_response_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="使用者說了這段話：'{user_input}'。請給出一段正向互動的回應。"
)
positive_response_chain = LLMChain(llm=model, prompt=positive_response_prompt)


def execute_conditional_chain(user_input):
    sentiment_result = sentiment_analysis_chain.run({"user_input": user_input})

    if sentiment_result.strip().lower() == "negative":
        return negative_response_chain.invoke({"user_input": user_input})
    else:
        return positive_response_chain.invoke({"user_input": user_input})

result = execute_conditional_chain("我對於你們的服務感到非常滿意，服務人員很用心，環境也很整潔。")

print(result["text"])
