from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
# import langchain
from langchain.chains import RetrievalQA

# langchain.debug = True

load_dotenv()
embeddings = OpenAIEmbeddings()

chat = ChatOpenAI()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
    # chain_type = "map_reduce"
    # chain_type = "map_rerank"
    # chain_type = "refine"
    # verbose = True
)

result = chain.run("What is an interesting fact about the English language ?")

print("\n\n")
print(result)