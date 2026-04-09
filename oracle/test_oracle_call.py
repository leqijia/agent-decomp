import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
payload = {
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [
        {"role": "user", "content": "Say hello and confirm you are Claude Sonnet."}
    ],
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]["content"]
    print("Response:", message)
except requests.exceptions.HTTPError as e:
    if response.status_code == 401:
        print("Error: Invalid API key. Check OPENROUTER_API_KEY in your .env file.")
    elif response.status_code == 402:
        print("Error: Insufficient credits on your OpenRouter account.")
    else:
        print(f"HTTP error {response.status_code}: {e}")
        print("Response body:", response.text)
    sys.exit(1)
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to OpenRouter. Check your internet connection.")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("Error: Request timed out after 30 seconds.")
    sys.exit(1)
except (KeyError, IndexError) as e:
    print(f"Error: Unexpected response format: {e}")
    print("Response body:", response.text)
    sys.exit(1)
