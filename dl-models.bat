@ECHO OFF
CD /d %~dp0

SET DOWNLOAD_MODEL_DIR=%~dp0models\pretrained
IF NOT EXIST "%DOWNLOAD_MODEL_DIR%" mkdir "%DOWNLOAD_MODEL_DIR%"


@REM download DPHuBERT models
SET DPHUBERT_DIR=https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPWavLM-sp0.75.pth
SET DPHUBERT_FILE=DPWavLM-sp0.75.pth
SET DPHUBERT_BIN=%DOWNLOAD_MODEL_DIR%\dphubert\%DPHUBERT_FILE%
IF NOT EXIST "%DPHUBERT_BIN%" (
    ECHO Info: Start downloading the DPHuBERT (DPWavLM) model...
    ECHO .
    curl -L -o "%DPHUBERT_BIN%" "%DPHUBERT_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the DPWavLM model...
        EXIT /b
    )
)

@REM download pyannote.audio ported wespeaker-voxceleb-resnet34 models
SET PYANNOTE_WESPEAKER_VOXCELEB_DIR=https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin
SET PYANNOTE_WESPEAKER_VOXCELEB_FILE=pytorch_model.bin
SET PYANNOTE_WESPEAKER_VOXCELEB_BIN=%DOWNLOAD_MODEL_DIR%\pyannote.audio\wespeaker-voxceleb-resnet34-LM\%PYANNOTE_WESPEAKER_VOXCELEB_FILE%
IF NOT EXIST "%PYANNOTE_WESPEAKER_VOXCELEB_FILE%" (
    ECHO Info: Start downloading the Pyannote.audio ported wespeaker-voxceleb-resnet34 model...
    ECHO .
    curl -L -o "%PYANNOTE_WESPEAKER_VOXCELEB_BIN%" "%PYANNOTE_WESPEAKER_VOXCELEB_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the Pyannote.audio ported wespeaker-voxceleb-resnet34 model...
        EXIT /b
    )
)

@REM download pyannote.audio ported wespeaker-voxceleb-resnet34 config
SET PYANNOTE_WESPEAKER_VOXCELEB_CONF_DIR=https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/config.yaml
SET PYANNOTE_WESPEAKER_VOXCELEB_CONF_FILE=config.yaml
SET PYANNOTE_WESPEAKER_VOXCELEB_CONF_BIN=%DOWNLOAD_MODEL_DIR%\pyannote.audio\wespeaker-voxceleb-resnet34-LM\%PYANNOTE_WESPEAKER_VOXCELEB_CONF_FILE%
IF NOT EXIST "%PYANNOTE_WESPEAKER_VOXCELEB_CONF_FILE%" (
    ECHO Info: Start downloading the Pyannote.audio ported wespeaker-voxceleb-resnet34 config...
    ECHO .
    curl -L -o "%PYANNOTE_WESPEAKER_VOXCELEB_CONF_BIN%" "%PYANNOTE_WESPEAKER_VOXCELEB_CONF_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the Pyannote.audio ported wespeaker-voxceleb-resnet34 config...
        EXIT /b
    )
)

@REM download RMVPE models (big thanks to yxlllc!!)
SET RMVPE_DIR=https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip
SET RMVPE_FILE=rmvpe.zip
SET RMVPE_BIN=%DOWNLOAD_MODEL_DIR%\rmvpe\%RMVPE_FILE%
IF NOT EXIST "%RMVPE_FILE%" (
    ECHO Info: Start downloading the RMVPE model...
    ECHO .
    curl -L -o "%RMVPE_BIN%" "%RMVPE_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the RMVPE model...
        EXIT /b
    )
)

@REM extract RMVPE models
SET RMVPE_PT_FILE=%DOWNLOAD_MODEL_DIR%\rmvpe\model.pt
IF NOT EXIST "%RMVPE_PT_FILE%" (
    ECHO.
    ECHO Info: Extract the RMVPE model...
    tar xf "%RMVPE_BIN%" -C "%DOWNLOAD_MODEL_DIR%\rmvpe"
    IF %ERRORLEVEL% neq 0 (
        ECHO.
        ECHO Error: Failed to extract the RMVPE model...
        EXIT /b
    )
)


@REM download MNP-SVC pretrained weights
@REM TODO: rewrite URLs
SET MNP_PTD_OPT0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/optimizer.bin
SET MNP_PTD_OPT1_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/optimizer_1.bin
SET MNP_PTD_MODEL0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/pytorch_model.bin
SET MNP_PTD_MODEL1_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/pytorch_model_1.bin
SET MNP_PTD_MODEL2_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/pytorch_model_2.bin
SET MNP_PTD_RND0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/random_states_0.pkl
SET MNP_PTD_SCHED0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/scheduler.bin
SET MNP_PTD_SCHED1_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/scheduler_1.bin
SET MNP_PTD_DIR=%DOWNLOAD_MODEL_DIR%\mnp-svc\states\cp0
IF NOT EXIST "%MNP_PTD_DIR%" (
    ECHO Info: Start downloading the MNP-SVC pretrained weights...
    ECHO .
    mkdir "%MNP_PTD_DIR%"
    CD /d "%MNP_PTD_DIR%"
    curl -L -O "%MNP_PTD_OPT0_DIR%"
    curl -L -O "%MNP_PTD_OPT1_DIR%"
    curl -L -O "%MNP_PTD_MODEL0_DIR%"
    curl -L -O "%MNP_PTD_MODEL1_DIR%"
    curl -L -O "%MNP_PTD_MODEL2_DIR%"
    curl -L -O "%MNP_PTD_RND0_DIR%"
    curl -L -O "%MNP_PTD_SCHED0_DIR%"
    curl -L -O "%MNP_PTD_SCHED1_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the MNP-SVC pretrained weights...
        EXIT /b
    )
    CD /d %~dp0
)

@REM download MNP-SVC weights pretrained on VCTK dataset
@REM TODO: rewrite URLs
@REM SET MNP_VCTK_OPT0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/optimizer.bin
@REM SET MNP_VCTK_OPT1_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/optimizer_1.bin
@REM SET MNP_VCTK_MODEL0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/pytorch_model.bin
@REM SET MNP_VCTK_MODEL1_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/pytorch_model_1.bin
@REM SET MNP_VCTK_MODEL2_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/pytorch_model_2.bin
@REM SET MNP_VCTK_RND0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/random_states_0.pkl
@REM SET MNP_VCTK_SCHED0_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/scheduler.bin
@REM SET MNP_VCTK_SCHED1_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/resolve/main/states/cp0/scheduler_1.bin
SET MNP_VCTK_MODEL_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/blob/main/model_10000/pytorch_model.bin
SET MNP_VCTK_CONF_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/blob/main/model_10000/config.yaml
SET MNP_VCTK_SPK_INFO_DIR=https://huggingface.co/TylorShine/MNP-SVC-v2-VCTK/blob/main/model_10000/spk_info.npz
SET MNP_VCTK_DIR=%DOWNLOAD_MODEL_DIR%\mnp-svc\vctk-full
IF NOT EXIST "%MNP_VCTK_DIR%" (
    ECHO Info: Start downloading the MNP-SVC weights pretrained on VCTK dataset...
    ECHO .
    mkdir "%MNP_VCTK_DIR%"
    CD /d "%MNP_VCTK_DIR%"
    curl -L -O "%MNP_VCTK_MODEL_DIR%"
    curl -L -O "%MNP_VCTK_CONF_DIR%"
    curl -L -O "%MNP_VCTK_SPK_INFO_DIR%"
    IF %ERRORLEVEL% neq 0 (
        ECHO .
        ECHO Error: Failed to download the MNP-SVC weights pretrained on VCTK dataset...
        EXIT /b
    )
    CD /d %~dp0
)


ECHO.
ECHO Info: All nececssary models downloaded successfully!
ECHO.
