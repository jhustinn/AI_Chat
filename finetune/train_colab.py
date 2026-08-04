# QLoRA Fine-Tuning Qwen2.5-1.5B untuk Customer Service
# Jalankan di Google Colab dengan GPU T4 (Runtime > Change runtime type > T4 GPU)

# ============================================================
# CELL 1: Install Dependencies (JALANKAN CELL INI DULU, LALU RESTART RUNTIME!)
# ============================================================
# !pip install -q --upgrade transformers peft bitsandbytes accelerate trl datasets pyarrow

# import os
# os._exit(0)  # Force restart runtime secara otomatis

# ============================================================
# CELL 2: Import Libraries (Jalankan SETELAH restart runtime!)
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset, Dataset
import json
import os

# Cek GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# CELL 3: Load Dataset
# ============================================================
# Upload dataset_example.json ke Colab, atau pakai cara ini:

# Option A: Upload dari lokal
# from google.colab import files
# uploaded = files.upload()  # Upload dataset_example.json

# Option B: Load dari Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# dataset_path = '/content/drive/MyDrive/dataset_example.json'

# Option C: Buat dataset langsung di sini
dataset_path = 'dataset_example.json'

with open(dataset_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"Jumlah data: {len(raw_data)}")

# ============================================================
# CELL 4: Format Dataset untuk Training
# ============================================================
def format_chat(example):
    """Format messages ke string untuk training"""
    messages = example['messages']
    formatted = ""
    for msg in messages:
        if msg['role'] == 'system':
            formatted += f"<|system|>\n{msg['content']}\n"
        elif msg['role'] == 'user':
            formatted += f"<|user|>\n{msg['content']}\n"
        elif msg['role'] == 'assistant':
            formatted += f"<|assistant|>\n{msg['content']}\n"
    return {"text": formatted}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_chat)
print("\nContoh format training:")
print(dataset[0]['text'][:500])

# ============================================================
# CELL 5: Konfigurasi QLoRA (4-bit Quantization)
# ============================================================
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model dengan quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Paksa float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare model untuk training
model = prepare_model_for_kbit_training(model)

print("Model loaded dengan 4-bit quantization!")

# ============================================================
# CELL 6: Konfigurasi LoRA Adapter
# ============================================================
lora_config = LoraConfig(
    r=16,                    # Rank (semakin tinggi = semakin banyak parameter)
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Layer yang akan di-train
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print jumlah parameter yang di-train
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
print(f"Total params: {all_params:,}")

# ============================================================
# CELL 7: Training Arguments
# ============================================================
output_dir = "./qwen2.5-cs-assistant-qlora"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,              # Jumlah epoch (3-5 untuk dataset kecil)
    per_device_train_batch_size=2,   # Batch size (sesuaikan dengan VRAM)
    gradient_accumulation_steps=4,   # Akumulasi gradient (effective batch = 2*4 = 8)
    learning_rate=2e-4,              # Learning rate
    weight_decay=0.01,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    fp16=False,                      # Disable fp16
    bf16=False,                      # Disable bf16
    optim="paged_adamw_8bit",        # Optimizer 8-bit (hemat VRAM)
    gradient_checkpointing=True,     # Hemat VRAM dengan checkpointing
    report_to="none",                # Nonaktifkan logging ke wandb
    max_grad_norm=0.3,
    seed=42,
)

# ============================================================
# CELL 8: Mulai Training!
# ============================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_column="text",
    max_seq_length=1024,             # Max panjang sequence
    tokenizer=tokenizer,
)

print("Mulai training...")
print("Estimasi waktu: 10-30 menit tergantung dataset")
trainer.train()

# ============================================================
# CELL 9: Simpan Model
# ============================================================
# Simpan hanya adapter (kecil, ~50MB)
adapter_dir = "./qwen2.5-cs-assistant-adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print(f"\nAdapter disimpan di: {adapter_dir}")

# List file yang dihasilkan
for f in os.listdir(adapter_dir):
    size = os.path.getsize(os.path.join(adapter_dir, f))
    print(f"  {f}: {size / 1024 / 1024:.2f} MB")

# ============================================================
# CELL 10: Download Adapter (PENTING!)
# ============================================================
# Kompres adapter menjadi zip
# !zip -r qwen2.5-cs-assistant-adapter.zip qwen2.5-cs-assistant-adapter/

# Download ke lokal
# from google.colab import files
# files.download('qwen2.5-cs-assistant-adapter.zip')

# ATAU simpan ke Google Drive
# import shutil
# shutil.copytree(adapter_dir, '/content/drive/MyDrive/qwen2.5-cs-assistant-adapter')

print("\n" + "="*50)
print("TRAINING SELESAI!")
print("="*50)
print("\nLangkah selanjutnya:")
print("1. Download adapter (zip)")
print("2. Ekstrak di laptop: E:\\App\\SWA\\AI\\finetune\\")
print("3. Jalankan script merge_adapter.py untuk gabungkan dengan base model")
