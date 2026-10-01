import httpx

def test():
    # Attempt to call the endpoint to see what it returns
    res = httpx.post("http://127.0.0.1:8000/api/login", json={"username": "admin", "password": "password"})
    if res.status_code != 200:
        print("Login failed:", res.text)
        return
    token = res.json()["access_token"]
    
    # Now call generate
    res2 = httpx.post(
        "http://127.0.0.1:8000/api/recipients/1/weekly-reports/generate?force=true",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0
    )
    print("Generate status:", res2.status_code)
    print("Content-Type:", res2.headers.get("content-type"))
    if res2.headers.get("content-type") == "application/json":
        print(res2.text[:500])

if __name__ == "__main__":
    test()
