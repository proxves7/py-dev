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


translate_template = """將以下中文文本翻譯成英文：
{input}"""

write_template = """根據以下提示創作一段文字：
{input}"""

general_template = """回答以下問題：
{input}"""

general_prompt = PromptTemplate(
    template=general_template,
    input_variables=["input"],
    output_variables=["text"]
)
general_chain = LLMChain(llm=model, prompt=general_prompt)

prompt_infos = [
    {
        "name": "translate_chain",
        "description": "進行中文翻譯成英文的任務",
        "prompt_template": translate_template,
    },
    {
        "name": "write_chain",
        "description": "進行創意寫作的任務",
        "prompt_template": write_template,
    },
]

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(model, router_prompt)
print("==========")
print(router_prompt)
print("==========")

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=model, prompt=prompt)
    destination_chains[name] = chain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=general_chain,
    verbose=True,
)

result = chain.invoke("請寫一篇關於那年夏天初戀的文章。");
# result = chain.invoke("翻譯這段話：10年前,我遇見了你,10年後,你遇見了我,於是我們一起遇見了彼此的未來。");
# result = chain.invoke("愛因斯坦是誰？");

print(result["text"])
