import logfire
from src.agents.state import AgentState
from src.config.config import settings
from langchain_groq import ChatGroq

llm=ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL,temperature=0)

def generate_node(state:AgentState):
    """
    Synthesize a response using both Documentation Context AND Conversation History
    """
    query=state["current_query"]

    history_str=""
    for msg in state["messages"][:-1]:
        role="User" if msg["role"]=="user" else "Assistant"
        history_str+=f"{role}: {msg['content']}\n"

    user_msg=state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory. ")
        prompt=f"""
        You are a friendly and helpful Enterprise AI Assitant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else:
        logfire.info("Generating technical RAG response.")
        max_content_chars=2500
        full_context=""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_content_chars:
                full_context+=doc+"\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt=f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT Provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSTAION HISTORY:
        {history_str}

        USER QUESTSION:
        "{user_msg}"
        """

    with logfire.span("LLM Synthesis"):
        try:
            content=llm.invoke(prompt).content
            logfire.info("Response synthesised via LLM.")

            return{
                "final_answer":content,
                "status":"Response generated.",
                "plan":state["plan"],
                "messages":[{"role":"assistant","content":content}]
            }
        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e