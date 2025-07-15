#!/bin/bash
cd ai_server
echo "Starting VitalFlow AI Server..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload