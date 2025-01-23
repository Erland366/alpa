import ray
import socket
from fastapi import FastAPI
from typing import Union, Optional, List, Tuple, Set, Dict
from enum import Enum
from datetime import datetime
from alpa.adaptdl.pollux_agent import PolluxAgent


class JobState(str, Enum):
    registered = "registered"
    queued = "queued"
    allocated = "allocated"
    started = "started"
    reallocating = "reallocating"
    ended = "ended"


class PolluxJob:
    id: str
    creation_time: datetime
    status: JobState
    pg_name: str
    pollux_agent: PolluxAgent
    
    def __init__(self, id: str):
        self.id = id
        self.creation_time = datetime.now()
        self.status = JobState.registered
        self.pollux_agent_jsonable = None
        self.pollux_agent = None


class ResourceReleaseReason(str, Enum):
    reallocation = "reallocation"
    ended = "ended"