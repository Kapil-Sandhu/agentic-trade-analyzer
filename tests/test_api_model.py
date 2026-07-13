import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the same configurations used in the orchestrator
load_dotenv(dotenv_path="config/.env")
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

print("=========================================")
print("🔍 FETCHING AVAILABLE GEMINI MODELS...")
print("=========================================\n")

try:
    models = client.models.list()
    for m in models.data:
        # We only care about the Gemini models, not the embedding ones
        if "gemini" in m.id.lower():
            print(f" ✅ {m.id}")
            
    print("\n=========================================")
    print("ACTION REQUIRED:")
    print("Copy one of the names above (e.g., 'gemini-1.5-flash')")
    print("and paste it into self.model_name in ai_orchestrator.py")
    print("=========================================")
    
except Exception as e:
    print(f"❌ API Error: {e}")