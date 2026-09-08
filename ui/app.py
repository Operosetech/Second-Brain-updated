import streamlit as st
import requests
import uuid
import time
import json

st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="wide")

st.title("🧠 Second Brain")
st.markdown("Agentic RAG Pipeline with LangGraph")

# Initialize session state for thread_id and chat history
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for config
with st.sidebar:
    st.header("Settings")
    st.text_input("Thread ID", value=st.session_state.thread_id, key="thread_id_input", disabled=True)
    if st.button("Reset Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []          
        st.rerun()

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plan"):
            with st.expander(f"✨ Thought Process ({len(msg['plan'])} steps)"):
                for step in msg["plan"]:
                    st.write(f"- {step}")
        if msg.get("sources"):
            st.markdown("**Sources:**")
            for source in msg["sources"]:
                st.info(source)

# React to user input
if prompt := st.chat_input("Ask anything about your knowledge base..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thought_placeholder = st.empty()
        
        try:
            with st.spinner("Connecting..."):
                response = requests.post(
                    "http://localhost:8000/stream",
                    json={"q": prompt, "thread_id": st.session_state.thread_id},
                    stream=True,
                    timeout=120
                )
                response.raise_for_status()
            
            answer = ""
            plan = []
            sources = []
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            if "token" in data_json:
                                answer += data_json["token"]
                                message_placeholder.markdown(answer + "▌")
                            elif "metadata" in data_json:
                                plan = data_json["metadata"].get("plan", plan)
                                sources = data_json["metadata"].get("sources", sources)
                            elif "error" in data_json:
                                st.error(data_json["error"])
                        except json.JSONDecodeError:
                            pass
            
            message_placeholder.markdown(answer if answer else "No response generated.")
            
            if plan:
                with st.expander(f"✨ Thought Process ({len(plan)} steps)"):
                    for step in plan:
                        st.write(f"- {step}")
            
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    st.info(source)
                    
            # Add to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "plan": plan,
                "sources": sources
            })
            
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
