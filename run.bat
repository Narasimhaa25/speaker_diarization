@echo off
echo.
echo ==================================================
echo   Speaker Diarization -- Starting...
echo ==================================================
echo.
echo   Open your browser at: http://localhost:5001
echo   Press Ctrl+C to stop.
echo.

set PYTHONPATH=speaker_diarization
venv\Scripts\python.exe ui/app.py
pause
