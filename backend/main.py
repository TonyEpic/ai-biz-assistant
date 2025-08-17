from fastapi import FastAPI

app = FastAPI(title="AI Biz Assistant (Day 1)")

@app.get("/ping")
def ping():
    return {"message": "pong"}
