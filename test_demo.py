import requests

data = {
  "user_id": 1,
  "text": "I feel shy talking to people but later I become comfortable",
  "screen_time": 5,
  "sleep": 6,
  "study": 2,
  "stress": 6,
  "date": "2026-03-30"
}

res = requests.post("http://127.0.0.1:5000/analyze", json=data)
print(res.json())