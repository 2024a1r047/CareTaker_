import sys
import os

sys.path.append('c:/CareTaker/backend')

from main import app

for route in app.routes:
    if hasattr(route, "methods") and "POST" in route.methods:
        if "weekly" in route.path:
            print(f"{route.path} -> {route.endpoint.__module__}.{route.endpoint.__name__}")
