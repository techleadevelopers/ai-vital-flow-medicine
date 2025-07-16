#!/usr/bin/env python3
"""
VitalFlow AI Server Startup Script
Starts the Python FastAPI ML server on port 8000
"""

import subprocess
import sys
import os

def main():
    print("🧠 Starting VitalFlow Advanced AI Server...")
    print("🔄 Initializing TensorFlow Neural Networks...")
    
    # Change to ai_server directory
    ai_server_dir = os.path.join(os.path.dirname(__file__), 'ai_server')
    
    if not os.path.exists(ai_server_dir):
        print("❌ AI server directory not found!")
        sys.exit(1)
    
    # Set TensorFlow environment variables for optimization
    env = os.environ.copy()
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    env['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
    
    try:
        # Start the FastAPI server with advanced ML models
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload",
            "--workers", "1"
        ]
        
        print(f"🔧 Running: {' '.join(cmd)}")
        print("📍 Advanced AI Server: http://localhost:8000")
        print("📊 Neural Network API docs: http://localhost:8000/docs")
        print("🚀 Ensemble models loading...")
        print("")
        
        # Run the server with TensorFlow optimizations
        subprocess.run(cmd, cwd=ai_server_dir, check=True, env=env)
        
    except KeyboardInterrupt:
        print("\n⏹️  AI Server stopped gracefully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting AI server: {e}")
        print("💡 Make sure TensorFlow and dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()