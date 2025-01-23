import alpa
from alpa.adaptdl.pollux_agent import pollux_agent
from typing import Union, Optional, List, Tuple, Set, Dict
import requests
import logging
import threading
import asyncio
import websockets
from datetime import datetime

def start_websocket_client():
    asyncio.run(websocket_client())

async def websocket_client():
    uri = f"ws://localhost:8000/ws?job_id={pollux_agent.job_id}"  # TODO: hardcoded
    print(f"Running websocket client")
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"Received message: {message}")
            asyncio.get_event_loop().call_soon_threadsafe(handle_message, message)

def handle_message(message):
    print(f"Handling message synchronously: {message} at time {datetime.now().strftime('%H:%M:%S')}")
    if message == "reallocation":
        pollux_agent.reallocation_approaching = True
    # if message == "command_xyz":
        # perform_action()