from langgraph_api.server import create_app

app = create_app()

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smith.langchain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)