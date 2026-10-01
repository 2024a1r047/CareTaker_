import sys
import time
sys.path.append('c:/CareTaker/backend')
from utils.gemini_client import call_gemini

def main():
    start = time.time()
    call_gemini({"contents": [{"parts": [{"text": "Hello"}]}]}, timeout=10, caller="[test]")
    print(f"Time taken: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
