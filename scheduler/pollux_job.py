import ray
import socket
from fastapi import FastAPI
from typing import Union, Optional, List, Tuple, Set, Dict
from enum import Enum
from datetime import datetime


class JobState(str, Enum):
    registered = "registered"
    queued = "queued"
    allocated = "allocated"
    started = "started"
    restarting = "restarting"
    ended = "ended"


class PolluxJob:
    id: str
    creation_time: datetime
    status: JobState
    pg_name: str
    
    def __init__(self, id: str):
        self.id = id
        self.creation_time = datetime.now()
        self.status = JobState.registered