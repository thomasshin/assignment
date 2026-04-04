from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from app.agent import solve

app = FastAPI()

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: Literal["A", "B", "C", "D"]

@app.post("/infer", response_model=AgentResponse)
def infer(req: AgentRequest):
    ans = solve(req.query)
    return AgentResponse(answer=ans)