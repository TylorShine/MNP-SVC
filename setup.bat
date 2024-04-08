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


@REM download pretrained models
SET MODELS_PRETRAINED_DIR=models\pretrained
@REM DPWavLM
SET DPWAVLM_FILE=https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPWavLM-sp0.75.pth
SET DPWAVLM_BIN=%MODELS_PRETRAINED_DIR%\dphubert\DPWavLM-sp0.75.pth
@REM pyannote.audio ported wespeaker-voxceleb-resnet34-LM
SET PYANNOTE_WESPEAKER_FILE=https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin
SET PYANNOTE_WESPEAKER_BIN=%MODELS_PRETRAINED_DIR%\pyannote.audio\wespeaker-voxceleb-resnet34-LM\pytorch_model.bin
SET PYANNOTE_WESPEAKER_CONFIG_FILE=https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/raw/main/config.yaml
SET PYANNOTE_WESPEAKER_CONFIG_YAML=%MODELS_PRETRAINED_DIR%\pyannote.audio\wespeaker-voxceleb-resnet34-LM\config.yaml
@REM RMVPE
SET RMVPE_FILE=https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip
SET RMVPE_ZIP=%DOWNLOAD_CACHE_DIR%\rmvpe.zip
SET RMVPE_DIR=%MODELS_PRETRAINED_DIR%\rmvpe
SET RMVPE_BIN=%RMVPE_DIR%\model.pt

IF NOT EXIST "%DPWAVLM_BIN%" (
    ECHO Info: Start downloading DPWavLM pre-trained model...
    ECHO.
    curl -L -o "%DPWAVLM_BIN%" "%DPWAVLM_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to download the DPWavLM pre-trained model
        EXIT /b
    )
)
IF NOT EXIST "%PYANNOTE_WESPEAKER_BIN%" (
    ECHO Info: Start downloading pyannote.audio ported wespeaker-voxceleb-resnet34-LM pre-trained model...
    ECHO.
    curl -L -o "%PYANNOTE_WESPEAKER_BIN%" "%PYANNOTE_WESPEAKER_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to download the pyannote.audio ported wespeaker-voxceleb-resnet34-LM pre-trained model
        EXIT /b
    )
)
IF NOT EXIST "%PYANNOTE_WESPEAKER_CONFIG_YAML%" (
    ECHO Info: Start downloading pyannote.audio ported wespeaker-voxceleb-resnet34-LM config file...
    ECHO.
    curl -L -o "%PYANNOTE_WESPEAKER_CONFIG_YAML%" "%PYANNOTE_WESPEAKER_CONFIG_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to download the pyannote.audio ported wespeaker-voxceleb-resnet34-LM config file
        EXIT /b
    )
)
IF NOT EXIST "%RMVPE_ZIP%" (
    ECHO Info: Start downloading RMVPE pre-trained model...
    ECHO.
    curl -L -o "%RMVPE_ZIP%" "%RMVPE_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to download the RMVPE pre-trained model
        EXIT /b
    )
)
IF NOT EXIST "%RMVPE_BIN%" (
    ECHO Info: Extract %RMVPE_ZIP%...
    ECHO.
    tar xf "%RMVPE_ZIP%" -C "%RMVPE_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the RMVPE pre-trained model zip
        EXIT /b
    )
)

@REM pre-trained model (VCTK)
SET MODELS_VCTK_DIR=models\vctk-partial
IF NOT EXIST "%MODELS_VCTK_DIR%" mkdir "%MODELS_VCTK_DIR%"
SET PRETRAINED_VCTK_FILE=https://huggingface.co/TylorShine/MNP-SVC-VCTK-partial/resolve/main/model_0.pt
SET PRETRAINED_VCTK_BIN=%MODELS_VCTK_DIR%\model_0.pt
SET PRETRAINED_VCTK_CONFIG_FILE=https://huggingface.co/TylorShine/MNP-SVC-VCTK-partial/raw/main/config.yaml
SET PRETRAINED_VCTK_CONFIG_YAML=%MODELS_VCTK_DIR%\config.yaml
SET PRETRAINED_VCTK_SPK_INFO_FILE=https://huggingface.co/TylorShine/MNP-SVC-VCTK-partial/resolve/main/spk_info.npz
SET PRETRAINED_VCTK_SPK_INFO_NPZ=%MODELS_VCTK_DIR%\spk_info.npz

IF NOT EXIST "%PRETRAINED_VCTK_BIN%" (
    ECHO Info: Start downloading MNP-SVC pre-trained model...
    ECHO.
    curl -L -o "%PRETRAINED_VCTK_BIN%" "%PRETRAINED_VCTK_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the MNP-SVC pre-trained model zip
        EXIT /b
    )
)
IF NOT EXIST "%PRETRAINED_VCTK_CONFIG_YAML%" (
    ECHO Info: Start downloading MNP-SVC pre-trained model config file...
    ECHO.
    curl -L -o "%PRETRAINED_VCTK_CONFIG_YAML%" "%PRETRAINED_VCTK_CONFIG_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the MNP-SVC pre-trained model config file
        EXIT /b
    )
)
IF NOT EXIST "%PRETRAINED_VCTK_SPK_INFO_NPZ%" (
    ECHO Info: Start downloading MNP-SVC pre-trained model spk_info...
    ECHO.
    curl -L -o "%PRETRAINED_VCTK_SPK_INFO_NPZ%" "%PRETRAINED_VCTK_SPK_INFO_FILE%"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the MNP-SVC pre-trained model spk_info
        EXIT /b
    )
)


ECHO.
ECHO Info: Ready
ECHO.
