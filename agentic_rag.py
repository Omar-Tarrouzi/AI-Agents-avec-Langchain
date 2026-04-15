from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain.tools import tool, BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Données pour l'exemple
chunks = [
    "je m'appelle Omar",
    "je suis étudiante en informatique",
    "j'aime regarder des films et j'adore la musique",
    "un de mes proverbes préféres: 'alone you go faster'"
]

# Initialize vector store
print("Initializing vector store with HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = Chroma.from_texts(
    texts=chunks,
    collection_name="cv_tformation",
    embedding=embeddings,
)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Nombre de documents à retourner
)
print(" Vector store initialized successfully!")

# Tool custom pour le retriever - VERSION CORRIGÉE
class RetrieverTool(BaseTool):
    name: str = "cv_tool"
    description: str = "Get information about me from my CV. Use this tool to answer questions about personal information, studies, hobbies, etc."

    def _run(self, query: str) -> str:
        """Synchronously retrieve relevant documents."""
        try:
            # Méthode 1: Utiliser invoke() (LangChain >= 0.1.0)
            docs = retriever.invoke(query)
            return "\n".join([doc.page_content for doc in docs])
        except AttributeError:
            try:
                # Méthode 2: Utiliser get_relevant_documents() (ancienne version)
                docs = retriever.get_relevant_documents(query)
                return "\n".join([doc.page_content for doc in docs])
            except AttributeError:
                # Méthode 3: Utiliser similarity_search directement
                docs = vector_store.similarity_search(query, k=3)
                return "\n".join([doc.page_content for doc in docs])

    async def _arun(self, query: str) -> str:
        """Asynchronously retrieve relevant documents."""
        return self._run(query)

retriever_tool = RetrieverTool()

# Tools
@tool
def get_employee_info(employee_id: str):
    """Get information about a given employee (name, salary, seniority)"""
    print(f" get_employee_info invoked with: {employee_id}")
    return {"name": employee_id, "salary": 12000, "seniority": "5"}

@tool
def send_email(email: str, subject: str, content: str):
    """Send an email with subject and content."""
    print(f" Email sent to {email} with subject '{subject}'")
    return f"Email sent to {email}"

tools = [get_employee_info, send_email, retriever_tool]

# LLM with Groq
print("Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=2,
)
llm_with_tools = llm.bind_tools(tools)
print(" Groq LLM initialized!")

# Agent node
def chatbot(state: State):
    print(f" Chatbot invoked with {len(state['messages'])} messages")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
print(" Building graph...")
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile graph
graph = graph_builder.compile()
print(f" Graph compiled successfully! Ready with {len(tools)} tools")