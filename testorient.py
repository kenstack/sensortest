# app.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root -> index.html
@app.get("/")
def root():
    return FileResponse("static/index2.html")

# Optional: simple health check
@app.get("/health")
def health():
    return {"ok": True}