# =============================================================================
# we will route between:
# 
# Web search: for questions related to recent events
# Self-corrective RAG: for questions related to our index
# =============================================================================

# =============================================================================
# Import Libraries
# =============================================================================
import getpass
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Annotated, Sequence, TypedDict, Literal


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# =============================================================================
# Load env variables
# =============================================================================
load_dotenv()

def _set_if_undefined(var:str):
    if var not in os.environ.keys():
        os.environ[var] = getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")


# =============================================================================
# Create Embedding & Chrome index & Retreiver
# =============================================================================
embd = OpenAIEmbeddings()


# Get some docs and add them to chroma db
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


# Splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore (Here you may get write access error, you can switch to proper vector databases such as pinecone)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)

# Create Retriever
retriever = vectorstore.as_retriever()

# =============================================================================
# =============================================================================
# # LLMS
# =============================================================================
# =============================================================================
# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    

# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
         ("system", system),
         ("human", "{question}")
     ]
    )

question_router = route_prompt | structured_llm_router

question_router.invoke({"question": "Who will the Bears draft first in the NFL draft?"})  # RouteQuery(datasource='vectorstore')

question_router.invoke({"question": "What are the types of agent memory?"})  # RouteQuery(datasource='vectorstore')









































































