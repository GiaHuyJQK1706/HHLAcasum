import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTester:
    """Class để test model tóm tắt văn bản"""
    
    def __init__(self, model_path: str = "./models/hhlai_academic_textsum"):
        """
        Khởi tạo model tester
        
        Args:
            model_path: Đường dẫn đến model fine-tuned
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🔧 Sử dụng device: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model không tìm thấy tại: {model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model đã fine-tune"""
        logger.info(f"📂 Đang tải model từ: {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            # Load với fp16 nếu có CUDA
            if torch.cuda.is_available():
                self.model = self.model.half()
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model đã load thành công\n")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi load model: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 50,
        num_beams: int = 4,
        language: str = "en",
    ) -> str:
        """
        Tóm tắt văn bản
        
        Args:
            text: Văn bản cần tóm tắt
            max_length: Độ dài tối đa của tóm tắt
            min_length: Độ dài tối thiểu của tóm tắt
            num_beams: Số beams cho beam search
            language: Ngôn ngữ ("en" hoặc "vi")
        
        Returns:
            Văn bản tóm tắt
        """
        # Thêm prefix ngôn ngữ
        prefix = "summarize: " if language == "en" else "tóm tắt: "
        input_text = prefix + text.strip()
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test model tóm tắt văn bản học thuật"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/hhlai_academic_textsum",
        help="Đường dẫn đến model fine-tuned"
    )
    
    parser.add_argument(
        "--language",
        choices=["en", "vi"],
        default="en",
        help="Ngôn ngữ (en: tiếng Anh, vi: tiếng Việt)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Độ dài tối đa của tóm tắt"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Độ dài tối thiểu của tóm tắt"
    )
    
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Số beams cho beam search"
    )
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo tester
        tester = ModelTester(model_path=args.model_path)
        
        # Hiển thị hướng dẫn
        lang_name = "Tiếng Anh" if args.language == "en" else "Tiếng Việt"
        print("=" * 80)
        print("🧪 HỆ THỐNG TEST TÓM TẮT VĂN BẢN HỌC THUẬT")
        print("=" * 80)
        print(f"📚 Ngôn ngữ: {lang_name}")
        print(f"⚙️  Max length: {args.max_length}, Min length: {args.min_length}")
        print(f"🔗 Model: {args.model_path}")
        print("=" * 80)
        print("\n💡 HƯỚNG DẪN:")
        print("  • Nhập văn bản cần tóm tắt")
        print("  • Nhấn Enter 2 lần để kết thúc nhập liệu")
        print("  • Nhập 'exit' để thoát chương trình")
        print("\n" + "=" * 80 + "\n")
        
        # Vòng lặp nhập và xử lý
        while True:
            # Nhập văn bản từ bàn phím
            print("📝 Nhập văn bản (Nhấn Enter 2 lần để hoàn thành):")
            print("-" * 80)
            
            lines = []
            empty_count = 0
            
            while empty_count < 2:
                try:
                    line = input()
                    if line.strip() == "":
                        empty_count += 1
                    else:
                        empty_count = 0
                        lines.append(line)
                except EOFError:
                    break
            
            text = " ".join(lines).strip()
            
            # Kiểm tra lệnh đặc biệt
            if text.lower() == "exit":
                print("\n👋 Cảm ơn đã sử dụng! Tạm biệt!")
                break
            
            if not text:
                print("⚠️  Văn bản không được để trống!\n")
                continue
            
            # Tóm tắt văn bản
            print("\n⏳ Đang tóm tắt...")
            try:
                summary = tester.summarize(
                    text=text,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    num_beams=args.num_beams,
                    language=args.language,
                )
                
                # Hiển thị kết quả
                print("\n" + "=" * 80)
                print("📄 VĂN BẢN GỐC:")
                print("-" * 80)
                print(text)
                print("\n" + "=" * 80)
                print("✅ TÓM TẮT:")
                print("-" * 80)
                print(summary)
                print("=" * 80 + "\n")
                
            except Exception as e:
                logger.error(f"❌ Lỗi khi tóm tắt: {e}")
                print("\n")
                continue
    
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Hãy chạy fine_tune_for_academic_text_summary.py trước!")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n👋 Đã dừng chương trình. Tạm biệt!")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()