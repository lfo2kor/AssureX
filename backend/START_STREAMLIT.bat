@echo off
echo ================================================================================
echo Test Automation - Admin Portal
echo ================================================================================
echo.
echo Starting Streamlit application...
echo.
echo The application will open in your default browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

REM Activate virtual environment and run streamlit
call venv\Scripts\activate.bat
streamlit run web_app.py

pause
