@echo off
echo ========================================
echo   Qwen2.5 CS Assistant - Fine-tuned
echo   Customer Service + Director Assistant
echo ========================================
echo.
echo Starting server on http://localhost:8000
echo.
E:\App\SWA\AI\venv\Scripts\python.exe -m llama_cpp.server ^
  --model E:\App\SWA\AI\models\qwen2.5-cs-assistant.gguf ^
  --n_ctx 2048 ^
  --n_gpu_layers 0 ^
  --chat_format chatml ^
  --host 0.0.0.0 ^
  --port 8000 ^
  --n_threads 6
