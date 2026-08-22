import logfire 
import traceback
import json
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import Optional

from src.agents.graph import rag_agent

logfire.configure(send_to_logfire="if-token-present", service_name="rag-project")

app = FastAPI(title="Enterprise Agentic RAG API")

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"

@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}

@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}

    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": ""
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        # rail_fired, rail_response = guard(q)
        # if rail_fired:
        #     logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
        #     return {
        #         "question": q,
        #         "answer": rail_response,
        #         "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
        #         "status": "Blocked by guardrails.",
        #         "sources": []
        #     }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        traceback.print_exc()
        logfire.exception(f"Backend execution failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }



@app.post("/stream")
async def stream_query(request: QueryRequest):
    """
    Streams LLM tokens via Server-Sent Events (SSE) as they are generated.
    The client reads the response body incrementally.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": ""
    }

    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # stream_mode="messages" yields (AIMessageChunk, metadata) tuples
            # as the LLM produces tokens
            async for chunk, metadata in rag_agent.astream(
                initial_state, config=config, stream_mode="messages"
            ):
                # chunk is an AIMessageChunk; only forward non-empty text
                token = getattr(chunk, "content", "")
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            traceback.print_exc()
            logfire.exception(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Signal the client that streaming is done
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import subprocess
    import sys
    import signal
    import os
    import uvicorn
    import threading

    UI_DIR = os.path.join(os.path.dirname(__file__), "ui")
    start_ui = "--no-ui" not in sys.argv

    ui_proc = None

    def start_vite():
        """Start the Vite dev server for the React UI."""
        global ui_proc
        # Prefer npx so it works even without a global npm install
        cmd = ["npm", "run", "dev"]
        ui_proc = subprocess.Popen(cmd, cwd=UI_DIR)
        ui_proc.wait()

    def shutdown(signum, frame):
        print("\n🛑  Shutting down both servers…")
        if ui_proc and ui_proc.poll() is None:
            ui_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if start_ui and os.path.isdir(UI_DIR):
        print("🖥️  Starting React UI  → http://localhost:5173")
        ui_thread = threading.Thread(target=start_vite, daemon=True)
        ui_thread.start()
    elif not os.path.isdir(UI_DIR):
        print("⚠️  No 'ui/' directory found — skipping UI. Run with --no-ui to suppress this warning.")

    print("🚀  Starting FastAPI backend → http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload="--no-reload" not in sys.argv)


# uv run python main.py          # starts both backend + UI
# uv run python main.py --no-ui  # starts only the FastAPI backend
# uv run python main.py --no-reload  # starts both, without uvicorn hot-reload