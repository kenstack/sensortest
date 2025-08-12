from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()  # ensure the variable is literally named "app"

# static files (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    idx = Path("static/index2.html")
    return FileResponse(idx) if idx.exists() else {"msg": "Hello from FastAPI"}
