"""
@ file model_setup.py: Model Setup Helper - Kiểm tra và setup model cục bộ
@ Copyright (C) 2025 by HHL Team
@ Update: Change model
"""
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def check_local_model(model_path: str = "./models/hhlai_acasum_t5_base") -> bool:
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    # Kiểm tra các file cần thiết
    required_files = [
        "config.json",
        "model.safetensors",  # hoặc pytorch_model.bin
        "tokenizer.model"     # hoặc tokenizer.json
    ]
    
    # Hoặc kiểm tra pytorch_model.bin thay vì model.safetensors
    alternative_files = ["pytorch_model.bin", "tokenizer.json"]
    
    files_found = []
    for file in (model_dir / file for file in required_files if (model_dir / file).exists()):
        files_found.append(file.name)
    
    # Kiểm tra alternative files
    for file in alternative_files:
        if (model_dir / file).exists():
            files_found.append(file)
    
    print(f"📁 Model directory: {model_path}")
    print(f"📄 Files found: {len(files_found)}")
    
    if (model_dir / "config.json").exists():
        print(f"✅ Model config found")
        return True
    else:
        print(f"❌ Model config not found (config.json missing)")
        return False


def setup_local_model(model_path: str = "./models/hhlai_acasum_t5_base") -> bool:
    model_dir = Path(model_path)
    
    # Tạo thư mục nếu không tồn tại
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Model directory ready: {model_path}")
    
    return True


def download_and_setup_model(
    model_name: str = "hhlai/hhlai_acasum_t5_base",
    model_path: str = "./models/hhlai_acasum_t5_base",
    force_download: bool = False
) -> bool:
    model_dir = Path(model_path)
    
    # Kiểm tra xem model đã tồn tại chưa
    if not force_download and check_local_model(model_path):
        print(f"✅ Model already exists locally: {model_path}")
        return True
    
    try:
        # Tạo thư mục
        setup_local_model(model_path)
        
        print(f"\n📥 Downloading model: {model_name}")
        print(f"📍 Saving to: {model_path}")
        print("⏳ This may take several minutes...")
        
        # Download tokenizer
        print("\n1️⃣ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        print(f"✅ Tokenizer saved to {model_path}")
        
        # Download model
        print("\n2️⃣ Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(model_path)
        print(f"✅ Model saved to {model_path}")
        
        print("\n✅ Model setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {str(e)}")
        return False


def verify_model_files(model_path: str = "./models/hhlai_acasum_t5_base") -> dict:
    model_dir = Path(model_path)
    
    info = {
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "files": [],
        "size_mb": 0
    }
    
    if not model_dir.exists():
        return info
    
    # List all files
    total_size = 0
    for file in model_dir.rglob("*"):
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            info["files"].append({
                "name": file.name,
                "relative_path": str(file.relative_to(model_dir)),
                "size_mb": round(size / (1024 * 1024), 2)
            })
    
    info["size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return info


def print_model_info(model_path: str = "./models/hhlai_acasum_t5_base"):
    """
    In thông tin chi tiết về model
    Args:
        model_path: Đường dẫn đến thư mục model
    """
    info = verify_model_files(model_path)
    
    print("\n" + "="*60)
    print("📊 MODEL INFORMATION")
    print("="*60)
    print(f"Path: {info['path']}")
    print(f"Exists: {'✅ Yes' if info['exists'] else '❌ No'}")
    print(f"Total Size: {info['size_mb']} MB")
    print(f"Files Count: {len(info['files'])}")
    
    if info['files']:
        print("\nFiles:")
        for file_info in sorted(info['files'], key=lambda x: x['size_mb'], reverse=True)[:10]:
            print(f"  - {file_info['name']}: {file_info['size_mb']} MB")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test script - chạy để kiểm tra model

    MODEL_PATH = "./models/hhlai_acasum_t5_base"
    MODEL_NAME = "hhlai/hhlai_acasum_t5_base"

    print("\n🔍 Checking model setup...\n")
    
    # 1. Kiểm tra model cục bộ
    print("1️⃣ Checking local model...")
    if check_local_model(MODEL_PATH):
        print("\n✅ Local model found!")
        print_model_info(MODEL_PATH)
    else:
        print("\n⚠️ Local model not found")
        print("\n2️⃣ Attempting to download model...")
        if download_and_setup_model(MODEL_NAME, MODEL_PATH):
            print("\n✅ Model downloaded successfully!")
            print_model_info(MODEL_PATH)
        else:
            print("\n❌ Failed to download model")
            