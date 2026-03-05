# start_backend.py
import os
import subprocess

# Use Render's $PORT or default 8000
port = os.environ.get("PORT", 8000)

# Start uvicorn immediately so Render detects port
subprocess.run(f"uvicorn main:app --host 0.0.0.0 --port {port}", shell=True)
