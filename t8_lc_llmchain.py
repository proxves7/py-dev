import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

prompt = PromptTemplate.from_template("Translate the following English text to zh-tw: {text}")

chain = LLMChain(llm=model, prompt=prompt, verbose=True)

result = chain.invoke({"text": "Hello, how are you?"})
print(result)
