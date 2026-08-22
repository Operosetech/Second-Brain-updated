from src.agents.state import AgentState
from src.config.config import settings
from langchain_groq import ChatGroq
import logfire

llm=ChatGroq(api_key=settings.GROQ_API_KEY,model=settings.GROQ_MODEL, temperature=0)

def planner_node(state:AgentState):
    """
    The planner determines if a search is needed based on the entire comversation
    """
    history=""
    for msg in state["messages"][:-1]:
        role="User" if msg["role"]=="user" else "Assistant"
        history+=f"{role}:{msg['content']}\n"

    user_message=state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting, a general knowledge question, or a question that can be answered using ONLY the conversation history above, respond with 'CONVERSATIONAL'.
    2. If it is a technical enterprise question about Kubernetes, Intel, Networking, or the indexed Odoo documentation that requires fresh documentation, output a refined search query.
    3. Do not create a search query for unrelated everyday questions like cooking, coffee, travel, jokes, or casual chat.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
