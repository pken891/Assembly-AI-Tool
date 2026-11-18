import streamlit as st
import os
import tiktoken
import litellm
from gateway import check_internet, load_config, get_available_models
from dotenv import load_dotenv

load_dotenv()

# Streamlit page config must be the first command
st.set_page_config(page_title="Hybrid AI Gateway", layout="wide")

# ==========================================
# 1. STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config_models" not in st.session_state:
    # Handles pathing whether run from root or src/
    config_path = "config.json" if os.path.exists("config.json") else "../config.json"
    st.session_state.config_models = load_config(config_path)
if "is_online" not in st.session_state:
    st.session_state.is_online = check_internet()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("⚙️ Gateway Settings")
    
    # Network status toggle
    status = "🟢 ONLINE" if st.session_state.is_online else "🔴 OFFLINE"
    st.markdown(f"**Status:** {status}")
    if st.button("Refresh Connection"):
        st.session_state.is_online = check_internet()
        st.rerun()

    st.divider()
    
    # Model Selection
    active_models = get_available_models(st.session_state.config_models, online=st.session_state.is_online)
    model_names = [m["display_name"] for m in active_models]
    
    if not model_names:
        st.error("No models available.")
        st.stop()
        
    selected_model_name = st.selectbox("Select Model:", model_names)
    selected_model = next(m for m in active_models if m["display_name"] == selected_model_name)

# ==========================================
# 3. MAIN CHAT INTERFACE
# ==========================================
st.title("Portable Hybrid AI Gateway")

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. TWO-STEP EXECUTION FLOW
# ==========================================
# STEP A: Capture input. Disables the input box if a prompt is already pending.
if prompt := st.chat_input("Enter your prompt...", disabled=bool(st.session_state.pending_prompt)):
    st.session_state.pending_prompt = prompt
    st.rerun()

# STEP B: Evaluate pending prompt and require user confirmation
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    
    # Analyze tokens and cost
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(prompt))
    cost = (token_count / 1000) * selected_model["input_cost_per_1k"]
    
    # Render the user's prompt visually
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Render the cost analysis block directly below the prompt
    if token_count > selected_model["max_context"]:
        st.error(f"⚠️ Warning: {token_count} tokens exceeds max context of {selected_model['max_context']}!")
    else:
        st.info(f"**Analysis:** {token_count} / {selected_model['max_context']} tokens | Estimated Cost: ${cost:.6f}")
    
    # Render execution buttons side-by-side
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("Execute", type="primary"):
            # 1. Commit prompt to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 2. Extract routing variables
            litellm_id = selected_model["litellm_id"]
            api_base = selected_model.get("api_base")
            api_key = os.getenv(selected_model.get("api_key_env"), "sk-local-bypass")
            
            # 3. Execute and stream response
            with st.chat_message("assistant"):
                try:
                    response = litellm.completion(
                        model=litellm_id,
                        messages=st.session_state.messages,
                        api_base=api_base,
                        api_key=api_key,
                        stream=True
                    )
                    # st.write_stream handles the generator chunking automatically
                    full_response = st.write_stream((chunk.choices[0].delta.content or "" for chunk in response))
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Execution Error: {e}")
            
            # Clear pending state and refresh UI
            st.session_state.pending_prompt = None
            st.rerun()
            
    with col2:
        if st.button("Cancel"):
            # Discard the prompt and unlock the input box
            st.session_state.pending_prompt = None
            st.rerun()