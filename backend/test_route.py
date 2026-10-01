import asyncio
import httpx
import uvicorn
import threading
from main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

async def test():
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    await asyncio.sleep(3) # wait for server to start
    
    async with httpx.AsyncClient() as client:
        # login
        res = await client.post("http://127.0.0.1:8000/api/login", json={"username": "admin", "password": "password"})
        if res.status_code != 200:
            print("Login failed:", res.text)
            return
        token = res.json()["access_token"]
        
        # generate
        res = await client.post(
            "http://127.0.0.1:8000/api/recipients/1/weekly-reports/generate?force=true",
            headers={"Authorization": f"Bearer {token}"}
        )
        print("Status code:", res.status_code)
        print("Content-Type:", res.headers.get("content-type"))
        if res.headers.get("content-type") == "application/json":
            print(res.text[:200])

if __name__ == "__main__":
    asyncio.run(test())
