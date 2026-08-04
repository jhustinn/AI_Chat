# Simpan GGUF ke Google Drive (di Colab)
from google.colab import drive
import shutil

drive.mount('/content/drive')

# Copy ke Google Drive
shutil.copy(
    '/content/qwen2.5-cs-assistant.gguf',
    '/content/drive/MyDrive/qwen2.5-cs-assistant.gguf'
)

print("File tersimpan di Google Drive!")
print("Download dari: https://drive.google.com/drive/my-drive")
