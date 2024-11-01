@ECHO OFF
CD /d %~dp0

SET DOWNLOAD_CACHE_DIR=%~dp0cache\downloads
IF NOT EXIST "%DOWNLOAD_CACHE_DIR%" mkdir "%DOWNLOAD_CACHE_DIR%"


@REM download Micromamba
@REM SET MAMBA_DIR=https://micro.mamba.pm/api/micromamba/win-64/latest
SET MAMBA_DIR=https://github.com/mamba-org/micromamba-releases/releases/download/1.5.10-0/micromamba-win-64.tar.bz2
SET MAMBA_FILE=micromamba.tar.bz2
SET MAMBA_BIN=%DOWNLOAD_CACHE_DIR%\%MAMBA_FILE%
IF NOT EXIST "%MAMBA_BIN%" (
    ECHO Info: Start downloading the Micromamba...
    ECHO .
    curl -L -o "%MAMBA_BIN%" "%MAMBA_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the Micromamba
        EXIT /b
    )
)

@REM extract Micromamba
SET MAMBA_ROOT_DIR=Library\bin
IF NOT EXIST "%DOWNLOAD_CACHE_DIR%\%MAMBA_ROOT_DIR%" (
    ECHO.
    ECHO Info: Extract the Micromamba...
    tar xf "%MAMBA_BIN%" -C "%DOWNLOAD_CACHE_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the Micromamba
        EXIT /b
    )
)

@REM make env
SET PYTHON_VERSION="python<3.12"
SET MAMBA_ROOT_PREFIX=%DOWNLOAD_CACHE_DIR%\%MAMBA_ROOT_DIR%
SET MICROMAMBA_BIN=%MAMBA_ROOT_PREFIX%\micromamba.exe
if exist "%MAMBA_ROOT_PREFIX%\Scripts\" (
  goto :ACTIVATE
)

"%MICROMAMBA_BIN%" shell hook -s cmd.exe -p "%MAMBA_ROOT_PREFIX%" -v

:ACTIVATE
@REM call mamba_hook.bat
@REM SET OLDPWD=%CD%
@REM CD /d "%MAMBA_ROOT_PREFIX%\Scripts"
@REM call activate.bat
@REM CD /d "%OLDPWD%"
REM "%MICROMAMBA_BIN%" shell hook -s cmd.exe -p "%MAMBA_ROOT_PREFIX%" -v
SET MAMBA_ENV_NAME=mambaenv
IF NOT EXIST "%MAMBA_ROOT_PREFIX%\envs\%MAMBA_ENV_NAME%" (
    ECHO.
    ECHO Info: Make env...
    ECHO %MICROMAMBA_BIN%
    ECHO %DOWNLOAD_CACHE_DIR%
    ECHO %MAMBA_ROOT_DIR%
    ECHO %PYTHON_VERSION%
    PUSHD "%MAMBA_ROOT_PREFIX%\Scripts"
    call activate.bat
    POPD
    micromamba --version
    micromamba create -y -n %MAMBA_ENV_NAME% %PYTHON_VERSION% uv -c conda-forge
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to create venv environment
        EXIT /b
    )
    ECHO.
    ECHO =======================================================================
    ECHO Info: Successful create python environment.
    ECHO       We need restart shell, launch same script again and ready to use!
    ECHO =======================================================================
    ECHO.
    PAUSE
    EXIT
)

@REM call scripts\make_env.bat

@REM activate env
ECHO.
ECHO Info: Activate env...
@REM "%MICROMAMBA_BIN%" activate %MAMBA_ENV_NAME%
@REM micromamba.exe activate %MAMBA_ENV_NAME%
PUSHD "%MAMBA_ROOT_PREFIX%\Scripts"
call activate.bat %MAMBA_ENV_NAME%
POPD
WHERE python


@REM update pip
ECHO.
ECHO Info: Update pip...
python -m pip install -U pip

@REM install torch
SET TORCH_INSTALL_CMD=uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
SET TORCH_PACKAGE=torch
uv pip show torch > NUL
IF %ERRORLEVEL% neq 0 (
    ECHO.
    ECHO Info: Install torch...
    %TORCH_INSTALL_CMD%
)

@REM install requirements
ECHO.
ECHO Info: Check install requirements...
uv pip install -r requirements/main.txt


ECHO.
ECHO Info: Ready
ECHO.
