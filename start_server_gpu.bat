@echo off
set PATH=E:\App\SWA\AI\venv\Lib\site-packages\nvidia\cuda_runtime\bin;%PATH%
echo Starting llama.cpp with GPU (GTX 1650)...
E:\App\SWA\AI\venv\Scripts\python.exe -m llama_cpp.server ^
  --model E:\App\SWA\AI\models\qwen2.5-cs-assistant.gguf ^
  --n_ctx 512 ^
  --n_gpu_layers 99 ^
  --n_threads 6 ^
  --host 0.0.0.0 ^
  --port 8000
