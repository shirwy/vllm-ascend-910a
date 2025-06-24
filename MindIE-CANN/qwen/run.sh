curl -v http://0.0.0.0:8001/v1/chat/completions \
-H 'Content-Type: application/json' \
-d \
'{
  "model": "/data/models/Qwen/Qwen3-8B",
  "temperature": 0.6,
  "max_tokens": 128,
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "你是谁？"
    }
  ]
}'
