# Guide: Fine-Tuning Qwen2.5-1.5B di Google Colab (GRATIS)

## Persiapan

### 1. Siapkan Google Account
- Pastikan punya akun Google (Gmail)

### 2. Siapkan Dataset
- File: `dataset_example.json` (sudah dibuat)
- Format: JSON array dengan messages
- Isi dengan data chatbot Anda sendiri (minimal 50-100 contoh)

---

## Langkah-Langkah

### Step 1: Buka Google Colab

1. Buka browser, kunjungi: **https://colab.research.google.com**
2. Klik **"New Notebook"** (kanan bawah)
3. Ganti nama notebook: `QLoRA-Finetune-Qwen2.5`

### Step 2: Aktifkan GPU T4 (GRATIS!)

1. Klik menu **Runtime** > **Change runtime type**
2. Pilih **T4 GPU** di dropdown "Hardware accelerator"
3. Klik **Save**
4. Cek GPU dengan menjalankan cell:

```python
!nvidia-smi
```

Output harusnya menunjukkan **Tesla T4** dengan **15GB VRAM**.

### Step 3: Upload Dataset

**Option A: Upload langsung (mudah)**
```python
from google.colab import files
uploaded = files.upload()  # Pilih dataset_example.json
```

**Option B: Mount Google Drive (recommended untuk file besar)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset dari Drive
!cp /content/drive/MyDrive/dataset_example.json /content/
```

### Step 4: Install Dependencies

Jalankan cell ini:
```python
!pip install -q transformers>=4.40.0 peft>=0.10.0 bitsandbytes>=0.43.0 accelerate>=0.28.0 trl>=0.8.0 datasets
```

### Step 5: Copy Training Script

1. Buka file `train_colab.py` yang sudah dibuat
2. Copy semua isi ke cell Colab (atau upload file dan import)
3. Jalankan semua cell satu per satu

**ATAU upload script langsung:**
```python
from google.colab import files
uploaded = files.upload()  # Upload train_colab.py

# Jalankan script
%run train_colab.py
```
SEKARANG
### Step 6: Tunggu Training Selesai

- Estimasi waktu: **15-30 menit** untuk 15 contoh data
- Lebih banyak data = lebih lama
- Pantau progress di output cell

**Tips:**
- Jangan tutup browser saat training
- Jangan klik "Disconnect" di Colab
- Jika timeout, Colab akan reconnect otomatis

### Step 7: Download Adapter

Setelah training selesai, download adapter:

```python
# Kompres adapter
!zip -r qwen2.5-cs-assistant-adapter.zip qwen2.5-cs-assistant-adapter/

# Download ke lokal
from google.colab import files
files.download('qwen2.5-cs-assistant-adapter.zip')
```

**ATAU simpan ke Google Drive:**
```python
import shutil
shutil.copytree(
    'qwen2.5-cs-assistant-adapter',
    '/content/drive/MyDrive/qwen2.5-cs-assistant-adapter'
)
```

---

## Setelah Download Adapter

### Step 8: Ekstrak di Laptop

1. Copy file `qwen2.5-cs-assistant-adapter.zip` ke:
   ```
   E:\App\SWA\AI\finetune\
   ```

2. Ekstrak (klik kanan > Extract Here)

3. Pastikan struktur folder:
   ```
   E:\App\SWA\AI\finetune\
   ├── qwen2.5-cs-assistant-adapter\
   │   ├── adapter_config.json
   │   ├── adapter_model.safetensors
   │   └── ...
   ├── dataset_example.json
   ├── train_colab.py
   └── merge_adapter.py
   ```

### Step 9: Merge Adapter dengan Base Model

Jalankan di terminal (di folder finetune):
```powershell
cd E:\App\SWA\AI\finetune
E:\App\SWA\AI\venv\Scripts\python.exe merge_adapter.py
```

Script akan:
1. Load base model Qwen2.5-1.5B
2. Load adapter hasil training
3. Merge menjadi model baru
4. Convert ke format GGUF

### Step 10: Pakai Model Fine-Tuned

1. Copy file `.gguf` ke folder models:
   ```
   E:\App\SWA\AI\models\qwen2.5-cs-assistant.gguf
   ```

2. Update `start_server.bat`:
   ```batch
   --model E:\App\SWA\AI\models\qwen2.5-cs-assistant.gguf
   ```

3. Jalankan server:
   ```
   E:\App\SWA\AI\start_server.bat
   ```

4. Test di Postman:
   ```json
   {
     "model": "qwen2.5-cs-assistant",
     "messages": [
       {"role": "system", "content": "Kamu adalah asisten customer service yang profesional."},
       {"role": "user", "content": "Produk saya rusak, mau refund."}
     ],
     "max_tokens": 200
   }
   ```

---

## Troubleshooting

### "Out of Memory" saat training
- Kurangi `per_device_train_batch_size` dari 2 ke 1
- Kurangi `max_seq_length` dari 1024 ke 512
- Kurangi `r` (LoRA rank) dari 16 ke 8

### Training terlalu lama
- Kurangi `num_train_epochs` dari 3 ke 1
- Kurangi jumlah data training

### Adapter tidak kompatibel
- Pastikan base model sama (Qwen2.5-1.5B-Instruct)
- Jangan ganti-ganti model saat training

### Colab disconnect
- Simpan adapter ke Google Drive secara berkala
- Gunakan fitur "Save to Drive" di Colab

---

## Tips untuk Dataset yang Bagus

### Jumlah Data
- **Minimal:** 50 contoh (untuk eksperimen)
- **Ideal:** 200-500 contoh
- **Optimal:** 1000+ contoh

### Variasi Data
- Berbagai skenario pertanyaan
- Berbagai cara user bertanya (formal, informal, singkat)
- Edge case (pertanyaan sulit, permintaan aneh)

### Kualitas Data
- Response harus konsisten
- Tone/style sama di semua contoh
- Tidak ada typo atau kesalahan

### Contoh Template yang Baik
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

---

## Update Dataset

Untuk menambah data baru:

1. Edit file `dataset_example.json`
2. Tambah object baru dengan format yang sama
3. Upload ke Colab
4. Jalankan ulang training (akan continue dari checkpoint)

---

## Referensi

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
