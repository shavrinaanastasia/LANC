@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo === War and Peace: lemmatisation + SVD embeddings (Gromov 2024 protocol) ===
rem Uses the base Anaconda, which already has natasha, numpy, scipy. Nothing is installed.
set "A=%USERPROFILE%\anaconda3"
set "PATH=%A%;%A%\Library\bin;%A%\Library\mingw-w64\bin;%A%\Scripts;%PATH%"
"%A%\python.exe" -c "import natasha, numpy, scipy; print('natasha', natasha.__file__)" || (echo natasha not found in Anaconda & pause & exit /b 1)
"%A%\python.exe" wap_preprocess.py --text "%USERPROFILE%\Downloads\war_and_peace_ru.txt" --out wap_embeddings.npz > preprocess_log.txt 2>&1
type preprocess_log.txt
if exist wap_embeddings.npz (echo. & echo DONE: wap_embeddings.npz created) else (echo. & echo FAILED, see preprocess_log.txt)
pause
