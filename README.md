# CriderGPT Engine

FastAPI backend for AI-driven chat with memory persistence.

## API Usage

### Chat with AI
`POST /chat-with-ai`

Example Request Payload:
```json
{
  "message": "Hello, how are you?",
  "user_id": "user_12345",
  "conversation_id": "conv_67890"
}
```

### Chat (Alternative)
`POST /chat`
Accepts the same payload as `/chat-with-ai`.
