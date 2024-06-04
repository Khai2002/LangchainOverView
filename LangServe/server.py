# LANGSERVE SERVER SIDE
# Create a server with FastAPI

from dotenv import load_dotenv
load_dotenv()

from fastapi import (FastAPI)
from langchain_openai import ChatOpenAI
from langserve import add_routes


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo-0125"),
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)