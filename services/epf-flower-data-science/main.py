import uvicorn
from fastapi.responses import RedirectResponse
from src.app import get_application
from src.schemas.message import MessageResponse
import pandas as pd
import os

app = get_application()

@app.get("/")
def redirect():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
