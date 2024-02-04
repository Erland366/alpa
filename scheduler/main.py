import ray
import socket
from fastapi import FastAPI, WebSocket, HTTPException
from typing import Union, Optional, List, Tuple, Set, Dict
from enum import Enum
import uvicorn
from contextlib import asynccontextmanager
from orchestrator import orchestrator
from pollux_job import PolluxJob
from pydantic import BaseModel
import pickle

HOST = "127.0.0.1"
PORT = 8000

    
@asynccontextmanager
async def lifespan(app: FastAPI):
    orchestrator.ip_address = f"{HOST}:{PORT}"
    yield
    
    
app = FastAPI(lifespan=lifespan)
    

class PlacementGroupRequest(BaseModel):
    num_hosts: int
    host_num_devices: List[int]
    name: str
    job_id: str
    
    
@app.post("/initial-request-placement-group")
async def initial_request_placement_group(job_id: str):
    try:
        orchestrator.initial_request_placement_group(job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.message)

    
@app.post("/create-placement-group")
async def create_placement_group(placementgrouprequest: PlacementGroupRequest):
    try:
        pg = orchestrator.create_placement_group(placementgrouprequest.num_hosts, placementgrouprequest.host_num_devices, placementgrouprequest.name,
                                                 placementgrouprequest.job_id)
        print(f"Placement group: {ray.util.get_placement_group(placementgrouprequest.name)}")
        return {"message": f"placement group with namespace {placementgrouprequest.name} created!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.message)
    
    
@app.post("/register-job")
async def register_job():
    job_id = orchestrator.register_job()
    return {"message": "succesfully registered job with id {job_id}", "job_id": job_id}


@app.get("/get-all-jobs")
async def get_all_jobs():
    jobs = orchestrator.get_all_jobs()
    return {"jobs": jobs}


# @app.post("/replace-all-jobs")
# async def replace_all_jobs(jobs: Dict[str, PolluxJob]):
#     orchestrator.replace_all_jobs(jobs)
#     return {"message": "successfully replaced all the existing jobs"}


if __name__=="__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="debug") # TODO: set reload to False