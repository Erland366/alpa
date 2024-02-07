import alpa
from alpa.adaptdl.pollux_agent import pollux_agent
from typing import Union, Optional, List, Tuple, Set, Dict
import requests
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def register_job() -> str:
    request = "/register-job"
    url = pollux_agent.scheduler_address + request
    r = requests.post(url=url)
    data = r.json()
    pollux_agent.job_id = data['job_id']
    logger.info(f"Registered job with orchestrator with job_id {pollux_agent.job_id}")
    

def register_placement_group(num_hosts: int, host_num_devices: List[int], name: str):
    request = "/create-placement-group"
    url = pollux_agent.scheduler_address + request
    request_data = {
        "num_hosts": num_hosts,
        "host_num_devices": host_num_devices,
        "name": name,
        "job_id": pollux_agent.job_id
    }
    r = requests.post(url=url, json=request_data)
    logger.info(f"Got placement group for job_id {pollux_agent.job_id}")
  
    
def release_resources():
    request = "/release-resources"
    url = pollux_agent.scheduler_address + request
    request_data = {
        "job_id": pollux_agent.job_id
    }
    r = requests.post(url=url, data=request_data)
    logger.info(f"Sent request to release the resources of job {pollux_agent.job_id}")
    