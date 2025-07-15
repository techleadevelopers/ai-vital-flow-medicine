#!/usr/bin/env python3
"""
VitalFlow AI Server Startup Script
Starts the Python FastAPI ML server on port 8000
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting VitalFlow AI Server...")
    
    # Change to ai_server directory
    ai_server_dir = os.path.join(os.path.dirname(__file__), 'ai_server')
    
    if not os.path.exists(ai_server_dir):
        print("❌ AI server directory not found!")
        sys.exit(1)
    
    try:
        # Start the FastAPI server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ]
        
        print(f"🔧 Running: {' '.join(cmd)}")
        print("📍 Server will be available at: http://localhost:8000")
        print("📊 API docs available at: http://localhost:8000/docs")
        print("")
        
        # Run the server
        subprocess.run(cmd, cwd=ai_server_dir, check=True)
        
    except KeyboardInterrupt:
        print("\n⏹️  AI Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting AI server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()