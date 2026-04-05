from fastapi import FastAPI
from app.schemas import AgentRequest, AgentResponse
from app.agent import run_agent

app = FastAPI()

@app.post("/infer", response_model=AgentResponse)
def infer(req: AgentRequest):
    answer = run_agent(req.query)
    return AgentResponse(answer=answer)