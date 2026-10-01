import httpx
import json

def test():
    # Attempt to call the endpoint to see what it returns
    # First, let's login as 'admin' or create a user to get a token.
    # Actually, we can just login.
    res = httpx.post("http://localhost:8000/api/login", json={
        "username": "admin",
        "password": "password"
    })
    if res.status_code != 200:
        print("Login failed:", res.text)
        return
    token = res.json()["access_token"]
    
    # Now call generate
    res2 = httpx.post(
        "http://localhost:8000/api/recipients/1/weekly-reports/generate?force=true",
        headers={"Authorization": f"Bearer {token}"}
    )
    print("Generate status:", res2.status_code)
    try:
        print(res2.json())
    except:
        print("Response is not JSON, length:", len(res2.content))

if __name__ == "__main__":
    test()
