import os
import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


def validate_order(order):
    """
    驗證訂單資訊
    """
    errors = []
    if not order.get("customer_id"):
        errors.append("缺少客戶ID")
    if not order.get("items") or len(order["items"]) == 0:
        errors.append("訂單中沒有商品")
    return {"order": order, "is_valid": len(errors) == 0, "errors": errors}

validate_order_RunnableLambda = RunnableLambda(validate_order)

def prepare_llm_input(processed_order):
    """
    準備 LLM 輸入
    """
    return {"order_info": json.dumps(processed_order, ensure_ascii=False)}

prepare_llm_input_RunnableLambda = RunnableLambda(prepare_llm_input)

summary_prompt= ChatPromptTemplate.from_template(
        "你是一個電子商務平台的客戶服務助手。請根據以下訂單內容生成訂單摘要。"
        "如果訂單無效，請解釋原因。訂單內容：### {order_info} ### "
    )
summary_chain =summary_prompt | model

workflow = RunnableSequence(
    validate_order_RunnableLambda,
    prepare_llm_input_RunnableLambda,
    summary_chain,
    StrOutputParser()
)

test_orders = [
    {
        "customer_id": "CUS001",
        "items": [
            {"name": "筆記本電腦", "price": 35000, "quantity": 1},
            {"name": "滑鼠", "price": 2500, "quantity": 2}
        ]
    },
    {
        "customer_id": "CUS002",
        "items": []
    },
    {
        "items": [
            {"name": "鍵盤", "price": 500, "quantity": 1}
        ]
    }
]

for order in test_orders:
    result = workflow.invoke(order)
    print(result)
    print("--------------------------------------------------")
