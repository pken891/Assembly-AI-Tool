import json
import os
import socket
import tiktoken
import litellm
from dotenv import load_dotenv

# ==========================================
# INITIALIZATION & ENVIRONMENT SETUP
# ==========================================
# Reads the hidden .env file at the project root and loads the API keys 
# into the system environment so os.getenv() can access them securely.
load_dotenv()

# ==========================================
# NETWORK & CONNECTIVITY TOGGLE
# ==========================================
def check_internet(host="8.8.8.8", port=53, timeout=1):
    """
    Pings a reliable server (Google DNS) to check for an active internet connection.
    Process: Attempts a fast, 1-second socket connection.
    Purpose: Used to determine if the gateway should expose cloud models and enable web search.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

# ==========================================
# CONFIGURATION LOADER
# ==========================================
def load_config(config_path="config.json"):
    """
    Reads the config.json file to get the list of available models.
    Process: Opens the JSON, parses the dictionary, and extracts the "models" array.
    Purpose: Acts as the centralized registry for both local endpoints and cloud APIs.
    """
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
            return config.get("models", [])
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Please create your configuration file.")
        return []

def get_available_models(models, online=True):
    """
    Filters the model list based on network connectivity.
    Process: Iterates through the models list. If offline, it drops models containing the "online" tag.
    Purpose: Prevents the user from attempting to route a prompt to an unreachable cloud API when offline.
    """
    if not online:
        return [m for m in models if "online" not in m.get("tags", [])]
    return models

# ==========================================
# EVALUATION & COST APPROXIMATION 
# ==========================================
def evaluate_prompt(prompt, models):
    """
    Calculates the token size of the prompt and estimates the execution cost.
    Process: Uses tiktoken to count tokens. Now displays both token cost and dollar cost 
             side-by-side, comparing the token count against the model's max context limit.
    Purpose: Prevents context window overflows and allows the user to see exact token consumption before routing.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(prompt))
    
    print(f"\n[Prompt Analyzed: {token_count} tokens]")
    print("-" * 50)
    
    for idx, model in enumerate(models):
        context_warning = ""
        if token_count > model["max_context"]:
            context_warning = " ⚠️ WARNING: EXCEEDS CONTEXT WINDOW"
            
        # Cost is calculated per 1,000 tokens based on config.json values
        estimated_cost = (token_count / 1000) * model["input_cost_per_1k"]
        
        print(f"[{idx + 1}] {model['display_name']}")
        # Displaying Token Cost vs Max Context explicitly
        print(f"    Token Cost:  {token_count} / {model['max_context']} max")
        
        # Displaying Dollar Cost
        if estimated_cost == 0:
            print(f"    Dollar Cost: $0.00 (Local Compute){context_warning}")
        else:
            print(f"    Dollar Cost: ${estimated_cost:.6f}{context_warning}")
            
    print("-" * 50)
    return token_count

# ==========================================
# EXECUTION & ROUTING
# ==========================================
def execute_prompt(prompt, model_config):
    """
    Routes the payload to the selected model using litellm.
    Process: Extracts API keys and URLs from the config, packages the prompt into standard JSON, 
             and streams the response chunks to the terminal.
    Purpose: Acts as the universal translator, handling the specific API requirements for local servers, Gemini, OpenAI, etc.
    """
    messages = [{"role": "user", "content": prompt}]
    
    litellm_id = model_config["litellm_id"]
    api_base = model_config.get("api_base")
    api_key_env = model_config.get("api_key_env")
    
    # Retrieve the actual key from the environment, fallback to a dummy key if local
    api_key = os.getenv(api_key_env, "sk-local-bypass")

    print(f"\n[Executing on {model_config['display_name']}...]")
    print("-" * 50)
    
    try:
        # litellm dynamically formats the payload and handles the request
        response = litellm.completion(
            model=litellm_id,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            stream=True
        )
        
        # Stream the text to the terminal in real-time
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"\nExecution Error: {e}")

# ==========================================
# MAIN INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    # 1. Connectivity Check
    is_online = check_internet()
    status_text = "ONLINE (Cloud & Web Search Active)" if is_online else "OFFLINE (Local Mode Only)"
    print(f"\n=== System Status: {status_text} ===")

    # 2. Load and filter models based on connectivity
    all_models = load_config()
    if not all_models:
        exit()
        
    active_models = get_available_models(all_models, online=is_online)
        
    # 3. Main Chat Loop
    while True:
        user_prompt = input("\nEnter prompt (or 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
            
        # 4. Web Search Intercept 
        if is_online and "@search" in user_prompt:
            print("\n[Scraping live data...]")
            try:
                import search_agent 
                search_query = user_prompt.replace("@search", "").strip()
                live_context = search_agent.perform_search(search_query) 
                user_prompt = f"Background web information:\n{live_context}\n\nUser Request: {search_query}"
            except ImportError:
                print("[Error] search_agent.py not found in the src/ directory. Bypassing search.")
            except AttributeError:
                print("[Error] Function 'perform_search' not found in search_agent.py. Update function name.")

        # 5. Analyze tokens and display detailed token/dollar cost
        evaluate_prompt(user_prompt, active_models)
        
        # 6. Capture selection, with a new option to revise the prompt if token cost is too high
        try:
            choice = input("\nSelect model number (or type 'edit' to revise prompt, Enter to cancel): ")
            
            if not choice.strip():
                continue
                
            if choice.lower() == 'edit':
                print("[Prompt discarded. Ready for new input.]")
                continue
                
            selection_index = int(choice) - 1
            if 0 <= selection_index < len(active_models):
                selected_model = active_models[selection_index]
                execute_prompt(user_prompt, selected_model)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")