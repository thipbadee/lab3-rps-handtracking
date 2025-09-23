# Lab 3 – Rock/Paper/Scissors with Hand Tracking

A minimal Flask app that uses MediaPipe Hands + OpenCV to play Rock/Paper/Scissors via webcam or uploaded image.

## Requirements
- Windows 10/11 recommended
- Python 3.12 (same as your venv) or Python 3.10–3.12
- A working webcam and permission to access it

## Setup (Windows, PowerShell)
1. Clone or copy this folder to your computer, open it in VS Code or a terminal.
2. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you see errors building packages, update build tools and pip:

```powershell
python -m pip install --upgrade pip
```

## Run the app

```powershell
python lab3-app.py
```

Then open your browser to:
- http://127.0.0.1:5000/

## Notes
- If the webcam doesn’t open, close any other apps using the camera and try again.
- On first run, MediaPipe may take a few seconds to initialize.
- To stop the server press `Ctrl+C` in the terminal.

## Troubleshooting
- If `opencv` fails to import, ensure your Python version matches the wheel (3.10–3.12 on Windows x64 is safe) and install the correct build.
- If ports are busy, change the port in `lab3-app.py` `app.run(..., port=5000)` to another free port.
