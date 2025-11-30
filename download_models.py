from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging
import os
import sys
import shutil

# Tắt logging để gọn gàng hơn
hf_logging.set_verbosity_warning()

# Danh sách models cần tải
MODELS = {
    # "facebook/bart-base": ("BART Base", "558MB"),
    # "t5-base": ("T5 Base", "892MB"),
    # "google/pegasus-xsum": ("PEGASUS XSum", "2.3GB"),
    # "allenai/led-base-16384": ("LED Base (long doc)", "500MB"),
    # "VietAI/vit5-base": ("VietAI viT5", "1.2GB"),
    "google/mt5-base": ("mT5 Base", "1.1GB"),
}

def print_progress(current, total, prefix='', suffix=''):
    """In thanh progress bar"""
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = 100 * current / total
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if current == total:
        print()

def download_model(model_name, display_name, size, current_num, total_num):
    """Tải 1 model về máy - TRỰC TIẾP vào ./models/"""
    print(f"\n{'='*60}")
    print(f"📥 [{current_num}/{total_num}] Đang tải: {display_name}")
    print(f"🔗 Model: {model_name}")
    print(f"💾 Kích thước: ~{size}")
    print(f"{'='*60}")
    
    try:
        # Tạo tên folder an toàn
        safe_name = model_name.replace("/", "_")
        # QUAN TRỌNG: Dùng đường dẫn tuyệt đối
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, "models", safe_name)
        
        # Tạo thư mục nếu chưa có
        os.makedirs(save_path, exist_ok=True)
        
        # Kiểm tra đã tải chưa
        config_file = os.path.join(save_path, "config.json")
        model_file = os.path.join(save_path, "pytorch_model.bin")
        
        if os.path.exists(config_file) and os.path.exists(model_file):
            print(f"⚠️  Model đã tồn tại tại {save_path}")
            choice = input("   Tải lại? (y/n): ")
            if choice.lower() != 'y':
                print("   ⏭️  Bỏ qua model này")
                return True
        
        # Phương pháp mới: Tải vào cache rồi MOVE sang ./models/
        print("⏳ [1/3] Đang tải tokenizer từ Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer hoàn thành")
        
        print("⏳ [2/3] Đang tải model weights từ Hugging Face...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("   ✅ Model weights hoàn thành")
        
        # Lưu TRỰC TIẾP vào ./models/ (không qua cache)
        print(f"⏳ [3/3] Đang lưu vào {save_path}...")
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("   ✅ Lưu file hoàn thành")
        
        # Kiểm tra kích thước thực tế
        total_size = 0
        for root, dirs, files in os.walk(save_path):
            for f in files:
                file_path = os.path.join(root, f)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        actual_size_mb = total_size / (1024 * 1024)
        
        print(f"✅ Đã lưu thành công!")
        print(f"   📂 Đường dẫn: {save_path}")
        print(f"   💾 Kích thước thực: {actual_size_mb:.1f}MB")
        
        # Giải phóng bộ nhớ
        del model
        del tokenizer
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Đã hủy bởi người dùng")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print(f"   💡 Tip: Kiểm tra kết nối mạng và thử lại")
        return False

def main():
    print("╔" + "="*58 + "╗")
    print("║" + " "*18 + "🚀 TẢI MODELS" + " "*27 + "║")
    print("╚" + "="*58 + "╝")
    
    # Hiển thị đường dẫn tuyệt đối
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    print(f"\n📂 Thư mục lưu: {models_dir}")
    print(f"📊 Số lượng models: {len(MODELS)}")
    
    # Hiển thị danh sách
    print("\n📋 Danh sách models sẽ tải:")
    for i, (model_name, (display_name, size)) in enumerate(MODELS.items(), 1):
        print(f"   {i}. {display_name:<30} ({size})")
    
    # Tính tổng kích thước
    total_mb = 0
    for _, (_, size) in MODELS.items():
        if "GB" in size:
            total_mb += float(size.replace("GB", "")) * 1024
        elif "MB" in size:
            total_mb += float(size.replace("MB", ""))
    
    print(f"\n💾 Tổng kích thước ước tính: ~{total_mb/1024:.2f}GB ({total_mb:.0f}MB)")
    print(f"⏱️  Thời gian ước tính: ~{len(MODELS)*5}-{len(MODELS)*15} phút (tùy mạng)")
    
    # Xác nhận
    print("\n" + "="*60)
    confirm = input("⚠️  Bạn có muốn tiếp tục? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Đã hủy")
        return
    
    # Tải từng model
    print("\n" + "="*60)
    print("🎯 BẮT ĐẦU TẢI")
    print("="*60)
    
    success_count = 0
    total_count = len(MODELS)
    
    for i, (model_name, (display_name, size)) in enumerate(MODELS.items(), 1):
        if download_model(model_name, display_name, size, i, total_count):
            success_count += 1
        
        # In progress tổng thể
        print_progress(i, total_count, prefix='📊 Tổng tiến độ:', suffix=f'{i}/{total_count} models')
    
    # Tổng kết
    print("\n\n" + "╔" + "="*58 + "╗")
    print("║" + " "*18 + "✨ HOÀN THÀNH" + " "*27 + "║")
    print("╚" + "="*58 + "╝")
    
    print(f"\n✅ Thành công: {success_count}/{total_count} models")
    
    if success_count < total_count:
        print(f"⚠️  Thất bại: {total_count - success_count} models")
        print("💡 Tip: Chạy lại script để tải các models bị lỗi")
    
    print(f"\n📂 Vị trí: {models_dir}")
    
    # Kiểm tra cấu trúc thư mục
    print("\n📁 Cấu trúc thư mục:")
    if os.path.exists(models_dir):
        items = sorted(os.listdir(models_dir))
        if items:
            for item in items:
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Đếm số file và tính kích thước
                    files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                    total_size = sum(
                        os.path.getsize(os.path.join(item_path, f)) 
                        for f in files
                    ) / (1024 * 1024)
                    print(f"   📦 {item}/ ({len(files)} files, {total_size:.1f}MB)")
        else:
            print("   (Trống)")
    
    # Hiển thị cách sử dụng
    print("\n" + "="*60)
    print("💡 CÁCH SỬ DỤNG:")
    print("="*60)
    print("from transformers import AutoTokenizer, AutoModelForSeq2SeqLM")
    print()
    print("# Load model từ local (KHÔNG CẦN MẠNG)")
    print("tokenizer = AutoTokenizer.from_pretrained('./models/facebook_bart-base')")
    print("model = AutoModelForSeq2SeqLM.from_pretrained('./models/facebook_bart-base')")
    print("="*60)
    
    # Thông tin cache (có thể xóa)
    print("\n💡 Lưu ý:")
    print("   - Models đã lưu trong ./models/")
    print("   - Cache tại ~/.cache/huggingface/ có thể XÓA được")
    print("   - Để xóa cache: rm -rf ~/.cache/huggingface/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng")
        sys.exit(0)
