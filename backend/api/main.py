from fastapi import FastAPI
from backend.api.router import agent_router

app = FastAPI(
    title="Data Whisperer API",
    description="API untuk analisis data dan interaksi dengan AI Agent."
)
app.include_router(agent_router.router)
@app.get("/")
def read_root():

    return {"status": "ok", "message": "Selamat datang di server Data Whisperer!"}