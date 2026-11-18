import json
import os
import tiktoken
import litellm
from dotenv import load_dotenv

# ==========================================
# INITIALIZATION & ENVIRONMENT SETUP
# ==========================================
# load_dotenv() reads the hidden .env file at the project root and loads 
# the API keys into the system environment so os.getenv() can access them securely.
load_dotenv()

# ==========================================
# CONFIGURATION LOADER
# ==========================================
def load_config(config_path="config.json"):
    """
    Reads the config.json file to get the list of available models.
    This acts as the registry for both local endpoints and cloud APIs.
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
    If offline (online=False), it removes any models tagged with "online" 
    so you don't accidentally try to route to Gemini without internet.
    """
    if not online:
        return [m for m in models if "online" not in m.get("tags", [])]
    return models

# ==========================================
# EVALUATION & COST APPROXIMATION
# ==========================================
def evaluate_prompt(prompt, models):
    """
    Calculates the exact token size of the raw prompt using tiktoken.
    It then iterates through the model registry, checks if the prompt exceeds 
    the model's max context window, and estimates the execution cost based 
    on the input rate defined in config.json.
    """
    # cl100k_base is the standard encoding used by most modern models (GPT-4, Claude, etc.)
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(prompt))
    
    print(f"\n[Prompt Analyzed: {token_count} tokens]")
    print("-" * 50)
    
    # Iterate through all available models to display a cost breakdown menu
    for idx, model in enumerate(models):
        context_warning = ""
        # Flag hardware or API limitations if the prompt is too large
        if token_count > model["max_context"]:
            context_warning = " ⚠️ WARNING: EXCEEDS CONTEXT WINDOW"
            
        # Calculate cost (rate is per 1,000 tokens)
        estimated_cost = (token_count / 1000) * model["input_cost_per_1k"]
        
        print(f"[{idx + 1}] {model['display_name']}")
        if estimated_cost == 0:
            print(f"    Cost: $0.00 (Local Compute){context_warning}")
        else:
            print(f"    Cost: ${estimated_cost:.6f}{context_warning}")
            
    print("-" * 50)
    return token_count

# ==========================================
# EXECUTION & ROUTING
# ==========================================
def execute_prompt(prompt, model_config):
    """
    Routes the payload to the selected model using litellm.
    litellm acts as a universal translator, taking our standard payload and 
    formatting it specifically for OpenAI, Gemini, Anthropic, or a local server.
    """
    messages = [{"role": "user", "content": prompt}]
    
    # Extract routing parameters from the selected model's config dictionary
    litellm_id = model_config["litellm_id"]
    api_base = model_config.get("api_base")
    api_key_env = model_config.get("api_key_env")
    
    # Fetch the actual API key from the environment. 
    # If it's a local model, it defaults to the dummy key "sk-local-bypass".
    api_key = os.getenv(api_key_env, "sk-local-bypass")

    print(f"\n[Executing on {model_config['display_name']}...]")
    print("-" * 50)
    
    try:
        # stream=True ensures the text prints to the terminal chunk-by-chunk 
        # as it is generated, rather than waiting for the entire response to finish.
        response = litellm.completion(
            model=litellm_id,
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            stream=True
        )
        
        # Process and print the streaming response
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
    # 1. Load the registry on startup
    all_models = load_config()
    if not all_models:
        exit()
        
    # 2. Enter the terminal chat loop
    while True:
        user_prompt = input("\nEnter prompt (or 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
            
        # 3. Analyze the token count and print the cost menu
        evaluate_prompt(user_prompt, all_models)
        
        # 4. Wait for user selection and handle input validation
        try:
            choice = input("\nSelect model number (or press Enter to cancel): ")
            if not choice.strip():
                continue # Skips execution and asks for a new prompt
                
            selection_index = int(choice) - 1
            if 0 <= selection_index < len(all_models):
                selected_model = all_models[selection_index]
                # 5. Route the prompt to the selected endpoint
                execute_prompt(user_prompt, selected_model)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")