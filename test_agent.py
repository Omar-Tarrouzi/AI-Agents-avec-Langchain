# test_agent.py
import requests
import json

# Your agent endpoint
BASE_URL = "http://127.0.0.1:2024"

# Create a thread
response = requests.post(f"{BASE_URL}/threads")
thread_id = response.json()['thread_id']
print(f"Thread created: {thread_id}")

# Send a message
payload = {
    "input": {
        "messages": [
            {"role": "user", "content": "What is my name?"}
        ]
    }
}

response = requests.post(
    f"{BASE_URL}/threads/{thread_id}/runs",
    json=payload
)
run_id = response.json()['run_id']
print(f"Run created: {run_id}")

# Get the result
response = requests.get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}/wait")
print("\nResponse:")
print(json.dumps(response.json(), indent=2))