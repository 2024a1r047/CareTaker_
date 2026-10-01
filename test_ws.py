import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:8000/api/ws/voice-chat") as ws:
        await ws.send(json.dumps({"text": "hello"}))
        res = await ws.recv()
        print(res)

asyncio.run(test())
