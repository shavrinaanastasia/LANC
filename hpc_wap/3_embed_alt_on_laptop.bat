@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo === War and Peace: CBOW + ruBERT embeddings (same vocabulary as the SVD run) ===
set "A=%USERPROFILE%\anaconda3"
set "PATH=%A%;%A%\Library\bin;%A%\Library\mingw-w64\bin;%A%\Scripts;%PATH%"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
"%A%\python.exe" -c "import natasha, gensim, torch, transformers; print('gensim', gensim.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" || (echo missing package & pause & exit /b 1)
"%A%\python.exe" -u wap_embed_alt.py --text "%USERPROFILE%\Downloads\war_and_peace_ru.txt" --svd wap_embeddings.npz --out wap_embeddings_alt.npz > embed_alt_log.txt 2>&1
type embed_alt_log.txt
if exist wap_embeddings_alt.npz (echo. & echo DONE: wap_embeddings_alt.npz created) else (echo. & echo FAILED, see embed_alt_log.txt)
pause
