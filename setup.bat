@ECHO OFF
CD /d %~dp0

SET DOWNLOAD_CACHE_DIR=cache\downloads
IF NOT EXIST "%DOWNLOAD_CACHE_DIR%" mkdir "%DOWNLOAD_CACHE_DIR%"


@REM download WinPython
SET WINPYTHON_DIR=https://github.com/winpython/winpython/releases/download/7.1.20240203final/
SET WINPYTHON_FILE=Winpython64-3.11.8.0dot.exe
SET WINPYTHON_BIN=%DOWNLOAD_CACHE_DIR%\%WINPYTHON_FILE%
IF NOT EXIST "%WINPYTHON_BIN%" (
    ECHO Info: Start downloading the WinPython...
    ECHO.
    curl -L -o "%WINPYTHON_BIN%" "%WINPYTHON_DIR%%WINPYTHON_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to download the WinPython
        EXIT /b
    )
)

@REM extract WinPython
SET WINPYTHON_ROOT_DIR=WPy64-31180
IF NOT EXIST "%DOWNLOAD_CACHE_DIR%\%WINPYTHON_ROOT_DIR%" (
    ECHO.
    ECHO Info: Extract the WinPython...
    "%WINPYTHON_BIN%" -o . -y
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the WinPython
        EXIT /b
    )
)

@REM make venv
SET WINPYTHON_VERSION=3.11.8
SET PYTHON_BIN=%DOWNLOAD_CACHE_DIR%\%WINPYTHON_ROOT_DIR%\python-%WINPYTHON_VERSION%.amd64\python.exe
IF NOT EXIST "venv" (
    ECHO.
    ECHO Info: Make venv...
    ECHO %PYTHON_BIN%
    ECHO %DOWNLOAD_CACHE_DIR%
    ECHO %WINPYTHON_ROOT_DIR%
    ECHO %WINPYTHON_VERSION%
    "%PYTHON_BIN%" -V
    "%PYTHON_BIN%" -m venv venv
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to create venv environment
        EXIT /b
    )
)

@REM activate venv
ECHO.
ECHO Info: Activate venv...
CALL venv\Scripts\activate
WHERE python


@REM update pip
ECHO.
ECHO Info: Update pip...
python -m pip install -U pip

@REM install torch
SET TORCH_INSTALL_CMD=pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
SET TORCH_PACKAGE=torch
pip show torch > NUL
IF %ERRORLEVEL% neq 0 (
    ECHO.
    ECHO Info: Install torch...
    %TORCH_INSTALL_CMD%
)

@REM install requirements
ECHO.
ECHO Info: Check install requirements...
pip install -r requirements\main.txt

ECHO.
ECHO Info: Ready
ECHO.
