import ray
import socket
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.param_functions import Query
from typing import Union, Optional, List, Tuple, Set, Dict
from enum import Enum
import uvicorn
from contextlib import asynccontextmanager
from orchestrator import orchestrator
from pollux_job import PolluxJob, ResourceReleaseReason
from pydantic import BaseModel
import pickle
import logging

HOST = "127.0.0.1"
PORT = 8000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

    
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
async def initial_request_placement_group(job_id: str, name: str):
    try:
        pg = await orchestrator.initial_request_placement_group(job_id, name)
        print(f"Placement group: {pg}")
        # print(f"Objects in the pg object: {dir(pg)}")
        return {"message": f"placement group with name {ray.util.placement_group_table(pg)['name']} created!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.message)

    
@app.post("/create-placement-group")
async def create_placement_group(placementgrouprequest: PlacementGroupRequest):
    try:
        pg = await orchestrator.create_placement_group(placementgrouprequest.num_hosts, placementgrouprequest.host_num_devices, placementgrouprequest.name,
                                                 placementgrouprequest.job_id)
        print(f"Placement group: {ray.util.get_placement_group(placementgrouprequest.name)}")
        return {"message": f"placement group with name {placementgrouprequest.name} created!"}
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

@app.get("/get-all-info")
async def get_all_jobs():
    info = orchestrator.get_all_info()
    return info

@app.post("/release-resources")
async def release_resources(job_id: str, reason: ResourceReleaseReason):
    try:
        orchestrator.release_resources(job_id, reason)
        return {"message": f"successfully released resources of job {job_id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.message)

@app.post("/update-state")
async def update_state(job_id: str = Query(...), state_bytes: UploadFile = File(...)):
    try:
        state_contents = await state_bytes.read()
        state_unpickled = pickle.loads(state_contents)
        orchestrator.update_state(job_id, state_unpickled)
        return {"message": f"successfully updated state of job {job_id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=e.message)

connected_clients: Dict[str, WebSocket] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    connected_clients[job_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Message from client: {data}")
    except Exception as e:
        del connected_clients[job_id]
        print(f"Client disconnected: {e}")


async def send_message_to_client(job_id: str, message):
    await connected_clients[job_id].send_json(message)


if __name__=="__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")