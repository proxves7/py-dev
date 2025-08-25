import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import init_chat_model
from dotenv import dotenv_values

config = dotenv_values(".env")
os.environ["GOOGLE_API_KEY"] = config.get("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_llm = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


#---------------------------------------------------------
loader = PyPDFLoader("resource/勞動基準法.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)

qdrant = QdrantVectorStore.from_documents(
    splits,
    embedding=embedding_llm,
    url="http://localhost:6333",
    collection_name="km_docs",
)
#---------------------------------------------------------


client = QdrantClient(url="http://localhost:6333")
collection_name = "km_docs"
qdrant = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embedding_llm
)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

q_template = ChatPromptTemplate.from_template("""你是一位精通台灣勞基法的專家。請根據以下參考資料回答問題：

參考資料：{context}

問題：{question}

專家回答：""")


qa_chain = (
    {
        "context": retriever ,
        "question": RunnablePassthrough(),
    }
    | q_template
    | model
    | StrOutputParser()
)


response = qa_chain.invoke("勞工加班費的計算方式是什麼？")

print(response)
