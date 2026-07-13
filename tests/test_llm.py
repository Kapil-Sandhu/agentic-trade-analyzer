import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load the Environment
load_dotenv(dotenv_path="config/.env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[ERROR] GEMINI_API_KEY not found in .env file!")
    exit()

# 2. The Model-Agnostic Routing Trick
# We use the OpenAI client, but explicitly point it to Google's servers.
print("[SYSTEM] Initializing LLM Client over Google OpenAI-Compatible Endpoint...")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def test_api_connection():
    print("[NETWORK] Sending test prompt to Gemini 2.5 Flash...")
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a quantitative trading AI. Be concise."},
                {"role": "user", "content": "What is the primary difference between a Maker and a Taker in order book dynamics? Limit to one sentence."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        # 3. Extract and display the response
        ai_message = response.choices[0].message.content
        print("\n==================================================")
        print("🟢 API CONNECTION SUCCESSFUL")
        print("==================================================")
        print(f"🤖 Gemini says: {ai_message.strip()}")
        print("==================================================\n")
        
    except Exception as e:
        print(f"\n[API ERROR] {e}\n")

if __name__ == "__main__":
    test_api_connection()