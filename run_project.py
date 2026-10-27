import os
import sys
import subprocess
import time
import webbrowser
import signal

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, 'backend')
    frontend_dir = os.path.join(root_dir, 'frontend')
    
    # Locate python executable in venv if present
    venv_python = os.path.join(root_dir, 'venv', 'Scripts', 'python.exe')
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    print("=" * 65)
    print("  AI-POWERED CARDIOVASCULAR DIGITAL TWIN — UNIFIED LAUNCHER")
    print("=" * 65)
    print(f"[1/3] Python Runtime: {python_exe}")
    print("[2/3] Launching FastAPI Backend on http://localhost:8000 ...")

    # Start backend
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=backend_dir
    )

    time.sleep(2)
    print("[3/3] Launching Vite Frontend on http://localhost:3000 ...")

    # Start frontend
    # Use npx or npm run dev
    shell_cmd = "npm run dev" if sys.platform != 'win32' else "npm.cmd run dev"
    frontend_proc = subprocess.Popen(
        shell_cmd,
        shell=True,
        cwd=frontend_dir
    )

    time.sleep(3)
    print("\n" + "=" * 65)
    print("  SERVICES RUNNING:")
    print("  - Frontend UI:  http://localhost:3000")
    print("  - Backend API:  http://localhost:8000 (Docs: http://localhost:8000/docs)")
    print("  - Health Check: http://localhost:8000/api/health")
    print("=" * 65)
    print("\nOpening web dashboard in your browser...")
    webbrowser.open("http://localhost:3000")
    print("\nPress Ctrl+C to stop all services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down services gracefully...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("All processes stopped.")

if __name__ == '__main__':
    main()
