import json
import os
import tiktoken
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()

def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
            return config.get("models", [])
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Please create your configuration file.")
        return []

def get_available_models(models, online=True):
    if not online:
        return [m for m in models if "online" not in m.get("tags", [])]
    return models

def evaluate_prompt(prompt, models):
    # cl100k_base is the standard encoding used by most modern models
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(prompt))
    
    print(f"\n[Prompt Analyzed: {token_count} tokens]")
    print("-" * 50)
    
    for idx, model in enumerate(models):
        context_warning = ""
        if token_count > model["max_context"]:
            context_warning = " ⚠️ WARNING: EXCEEDS CONTEXT WINDOW"
            
        estimated_cost = (token_count / 1000) * model["input_cost_per_1k"]
        
        print(f"[{idx + 1}] {model['display_name']}")
        if estimated_cost == 0:
            print(f"    Cost: $0.00 (Local Compute){context_warning}")
        else:
            print(f"    Cost: ${estimated_cost:.6f}{context_warning}")
            
    print("-" * 50)
    return token_count

if __name__ == "__main__":
    all_models = load_config()
    if not all_models:
        exit()
        
    # Test loop for Phase 2
    while True:
        user_prompt = input("\nEnter prompt (or 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
            
        evaluate_prompt(user_prompt, all_models)