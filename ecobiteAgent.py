from typing import Annotated, TypedDict, Sequence, Literal
from dataclasses import dataclass, field
from datetime import datetime
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma


from prompts.main_reply_prompt import get_main_reply_prompt
from helper.supabase import get_user_inventory

load_dotenv()
db_dir = "./chroma_db"
collection_name = "ecobite_faq"

# --- CONFIGS ---
@dataclass
class AgentConfig:
    """Centralized configuration"""
    model_name: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float = 0.5

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    config: AgentConfig

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Initialize Vector Store if available
vectorStore = None
retriever = None

if os.path.exists(db_dir) and os.listdir(db_dir):
    print(f"--- Existing database found in {db_dir}. Loading... ---")
    try:
        vectorStore = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        retriever = vectorStore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not load Chroma DB: {e}")

# --- TOOLS ---
def create_tools(config: AgentConfig): 
    @tool(description=
        """
        Use this tool ONLY when the user explicitly asks for the 'current date', 'current time', 
        'today's date', or 'what day is it'.
        """)
    def current_dateTime():
        now = datetime.now()
        return now.strftime("%b %d, %Y %H:%M")

    @tool(description=        
        """
        This tool searches and returns the information from the EcoBite FAQ document 
        to answer questions related to the app, features, donations, and usage.
        """)
    def ecobite_faq_retriever(query: str) -> str:
        if not retriever:
            return "FAQ database is currently unavailable."
            
        docs = retriever.invoke(query)

        if not docs:
            return f"Didn't find relevant information about {query} in the EcoBite FAQ document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Source Document Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    @tool(description=
        """
        Retrieves the user's current inventory, including item names, quantities, and expiration dates.
        Use this tool when the user asks about:
        - what items they currently have
        - what is expiring or near expiration
        - checking their stock or available ingredients
        - recipes or meal ideas based on inventory

        Requires: user_id
        """)
    def user_inventory_retriever(user_id: int) -> str:
        # Call the Supabase function
        items = get_user_inventory(user_id)
        
        if not items:
            return "The user's inventory is empty."
        
        # Format the output so the LLM can read it easily
        # Based on your table schema: item_name, quantity, unit, expiry_date
        formatted_list = ["Current Inventory:"]
        for item in items:
            name = item.get("item_name", "Unknown")
            qty = item.get("quantity", 0)
            unit = item.get("unit", "")
            expiry = item.get("expiry_date", "N/A")
            formatted_list.append(f"- {name}: {qty} {unit} (Expires: {expiry})")
            
        return "\n".join(formatted_list)

    return [current_dateTime, ecobite_faq_retriever, user_inventory_retriever]

# --- GRAPH BUILDER ---
def build_agent_graph(config: AgentConfig = None, checkpointer = None):
    if config is None:
        config = AgentConfig()

    # 1. Setup Tools and Model
    tools = create_tools(config)
    model = ChatOpenAI(
        model_name=config.model_name,
        temperature=config.temperature
    ).bind_tools(tools)

    # 2. Define Agent Node
    def agent_node(state: AgentState):
        messages = state["messages"]
        
        # Ensure System Prompt is always present
        # Note: You might need to update your main_reply_prompt to mention the new inventory capability
        system_prompt_content = get_main_reply_prompt(tools[1].name, tools[2].name)
        system_prompt = SystemMessage(content=system_prompt_content)
        
        # Filter out previous system messages to avoid stacking them
        filtered_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        all_messages = [system_prompt] + filtered_messages
        
        response = model.invoke(all_messages)
        return {"messages": [response]}

    # 3. Define Routing Logic
    def route_agent(state: AgentState) -> Literal["tools", "end"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "tools"
        return "end"

    # 4. Build the Graph
    graph = StateGraph(AgentState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=tools))

    graph.add_edge(START, "agent")
    
    graph.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "agent")

    # 5. Persistence Strategy
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)