# Yuki AI (Refactored)

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Test
```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"session_id":"demo","user_text":"Salom Yuki!"}'
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"session_id":"demo","user_text":"Menga motivatsiya ber!"}'
```
