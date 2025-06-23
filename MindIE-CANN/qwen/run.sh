curl -v http://0.0.0.0:3025/v1/chat/completions \
-H 'Content-Type: application/json' \
-d \
'{
  "model": "Qwen3-0.6B",
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
