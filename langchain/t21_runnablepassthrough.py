import os
import json
import time
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


style_examples = """
1. 一鄉二里，共三夫子不識四書五經六義，竟敢教七八九子，十分大膽
2. 十室九貧，湊得八兩七錢六分五毫四厘，尚且又三心二意，一等下流
3. 圖畫裡，龍不吟，虎不嘯，小小書童可笑可笑
4. 棋盤裡，車無輪，馬無韁，叫聲將軍提防提防
5. 鶯鶯燕燕翠翠紅紅處處融融洽洽
6. 雨雨風風花花葉葉年年暮暮朝朝
"""

writing_template = ChatPromptTemplate.from_template("""
你是一位精通對聯創作的文學大師。請根據以下提供的主題創作一組對聯。

主題: {topic}

請參考以下的寫作風格範例，創作時要體現類似的韻律感和文字技巧：

{style_examples}

要求：
1. 創作一組對仗工整、意境深遠的對聯
2. 對聯應與給定主題相關
3. 儘量融入範例中展現的數字遞進、重複疊字等修辭技巧
4. 確保對聯在音律和結構上和諧統一

請提供：
- 上聯
- 下聯
- 簡短解釋（說明對聯與主題的關聯，以及使用的技巧）
""")


couplet_generation_system = RunnableSequence(
    {
        "topic": RunnablePassthrough(),
        "style_examples": lambda _: style_examples
    },
    writing_template,
    model
)

# 使用對聯生成系統
result = couplet_generation_system.invoke({"topic": "生成式AI"})
print(result.content)
