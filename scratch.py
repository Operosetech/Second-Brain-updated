import asyncio
from src.agents.graph import rag_agent
import json

async def test_stream():
    initial_state = {
        "messages": [{"role": "user", "content": "What is LangGraph?"}],
        "current_query": "What is LangGraph?",
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": ""
    }
    
    config = {"configurable": {"thread_id": "test_thread"}}
    
    async for event in rag_agent.astream(
        initial_state, config=config, stream_mode=["messages", "values"]
    ):
        mode, data = event
        if mode == "messages":
            chunk, metadata = data
            token = getattr(chunk, "content", "")
            if token:
                print(f"TOKEN: {token}")
        elif mode == "values":
            plan = data.get("plan", [])
            docs = data.get("documents", [])
            print(f"METADATA: plan={len(plan)} steps, docs={len(docs)}")

if __name__ == "__main__":
    asyncio.run(test_stream())
