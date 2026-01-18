import requests

url = "http://127.0.0.1:8000/chat"
payload = {
    "query": "Explain binary search and implement it in Python.",
    "max_new_tokens": 400
}

res = requests.post(url, json=payload, timeout=300)
print("Status:", res.status_code)
print("Response:\n")
print(res.json()["response"][:1500])
