@echo off
chcp 65001 >nul
echo.
echo =====================================================
echo   ULTRA-FAST COLLEGE ATTENDANCE SYSTEM v0.1.5
echo     Built by Anshu Gondi – 1st Semester Legend
echo =====================================================
echo.

:: Go to the folder where start.bat is located
cd /d "%~dp0"
echo Working folder: %cd%
echo.

:: Activate venv (same level as start.bat)
echo Activating virtual environment...
call venv\Scripts\activate.bat 2>nul

:: If venv is missing → create it automatically
if %errorlevel% neq 0 (
    if not exist venv (
        echo Creating new virtual environment...
        python -m venv venv
        call venv\Scripts\activate.bat
    )
)

:: Upgrade pip
echo.
echo Installing/upgrading all required packages...
python -m pip install --upgrade pip >nul

:: INSTALL RUST WHEEL — FIXED FOREVER (no more wildcard error)
echo Installing ultra-fast Rust core...
for %%i in ("attendance_backend\rust_extensions\rust_backend\dist\rust_backend-*.whl") do (
    python -m pip install "%%i" --force-reinstall >nul
)

:: Install Django + all Python packages
python -m pip install django pillow opencv-python requests django-htmx >nul 2>&1
python -m pip install -r attendance_backend\requirements.txt >nul 2>&1

:: GO INTO DJANGO PROJECT
cd attendance_backend

echo.
echo Running database migration...
python manage.py migrate --noinput

echo.
echo =====================================================
echo               SYSTEM IS NOW LIVE!
echo =====================================================
echo → On this PC:           http://localhost:8000
echo → On phones (same Wi-Fi): http://%computername%.local:8000
echo     or open http://YOUR_IP:8000 (IP shown below)
echo.
echo Close this window to stop the system.
echo.
python manage.py runserver 0.0.0.0:8000

pause