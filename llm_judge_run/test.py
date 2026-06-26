import time
from groq import Groq

# The 5 keys you want to test
API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
]

MODEL = "llama-3.3-70b-versatile"

def test_keys():
    for i, key in enumerate(API_KEYS):
        print(f"Testing Key {i+1}/{len(API_KEYS)}: {key[:10]}...")
        try:
            client = Groq(api_key=key)
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "Say 'Key is working'"}
                ],
                max_tokens=20,
            )
            
            result = response.choices[0].message.content
            print(f"✅ Success: {result}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
        
        print("-" * 30)
        # Small sleep to avoid hitting rate limits immediately during testing
        time.sleep(1)

if __name__ == "__main__":
    test_keys()