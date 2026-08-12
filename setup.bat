@echo off
setlocal enabledelayedexpansion

:: --- Intelligent Driver Assistance Systems (IDAS) Setup Script ---
:: This script automates the environment setup and runs the application for Windows.

echo Starting IDAS Setup...

:: 1. Check for facial landmarks model
if not exist "shape_predictor_68_face_landmarks.dat" (
    echo Facial landmarks model missing.
    echo Please download it from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
    echo Extract it and place 'shape_predictor_68_face_landmarks.dat' in this directory.
    pause
    exit /b
)

:: 2. Create Virtual Environment
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment. Ensure Python is installed and in your PATH.
    pause
    exit /b
)

:: 3. Activate and Install Dependencies
echo Installing dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Run the Application
echo Setup complete! Starting IDAS...
python main.py

pause
