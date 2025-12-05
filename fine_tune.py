import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
import numpy as np
import logging
import time
from datetime import timedelta

# ============================================================================
# SETUP CUDA TỐI ƯU CHO RTX 3050 Ti LAPTOP WINDOWS 11
# By HHL Team
# ============================================================================
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Tắt warning về torch extensions
os.environ['TORCH_ALLOW_TF32'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Tắt distributed warnings
os.environ['NCCL_DEBUG'] = 'INFO'

torch.cuda.empty_cache()

# Tắt cuDNN benchmarking để ổn định
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Setup logging (chỉ INFO để bỏ DEBUG và WARNING dư thừa)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tắt warning từ transformers và torch
import warnings
warnings.filterwarnings('ignore')

# Tắt verbose logs từ transformers
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

class ProgressAndNaNCallback(TrainerCallback):
    """Callback để theo dõi tiến độ chi tiết và phát hiện/xử lý NaN"""
    
    def __init__(self):
        self.start_time = None
        self.loss_history = []
        self.nan_count = 0
        self.last_log_step = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Gọi ở đầu training"""
        self.start_time = time.time()
        self.nan_count = 0
        self.loss_history = []
        print("\n" + "=" * 80)
        print("🚀 BẮT ĐẦU HUẤN LUYỆN")
        print("=" * 80 + "\n")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Kiểm tra NaN và hiển thị tiến độ sau mỗi step"""
        if not state.log_history:
            return control
        
        last_log = state.log_history[-1]
        loss = last_log.get('loss')
        learning_rate = last_log.get('learning_rate', 0)
        
        # Kiểm tra NaN/Inf trong loss
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f"\n❌ ERROR: NaN/Inf in loss: {loss}")
            control.should_training_stop = True
            return control
        
        # Lưu loss
        if loss is not None:
            self.loss_history.append(loss)
        
        # Hiển thị tiến độ (mỗi 50 steps)
        if state.global_step % 50 == 0 or state.global_step == 1:
            total_steps = state.max_steps
            progress = state.global_step / total_steps if total_steps > 0 else 0
            
            # Tính thời gian còn lại
            elapsed = time.time() - self.start_time
            if progress > 0.01:
                eta_seconds = (elapsed / progress) - elapsed
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "Calculating..."
            
            # Progress bar
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            # Print
            print(f"⏳ [{state.global_step:5d}/{total_steps}] {bar} "
                  f"{progress*100:5.1f}% | Loss: {loss:.4f} | ETA: {eta_str}")
        
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Gọi ở cuối mỗi epoch"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
        
        print(f"\n{'='*80}")
        print(f"✅ EPOCH {int(state.epoch)} HOÀN THÀNH")
        print(f"{'='*80}")
        print(f"   ⏱️  Thời gian: {elapsed_str}")
        print(f"   📊 Step: {state.global_step}/{state.max_steps}")
        print(f"   📈 Loss: {self.loss_history[-1]:.4f} (avg: {avg_loss:.4f})")
        print(f"{'='*80}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Gọi ở cuối training"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        final_loss = self.loss_history[-1] if self.loss_history else 0
        min_loss = min(self.loss_history) if self.loss_history else 0
        
        print(f"\n{'='*80}")
        print(f"🎉 HUẤN LUYỆN HOÀN THÀNH!")
        print(f"{'='*80}")
        print(f"   ⏱️  Tổng thời gian: {elapsed_str}")
        print(f"   📊 Tổng steps: {state.global_step}")
        print(f"   📈 Final loss: {final_loss:.4f}")
        print(f"   📊 Min loss: {min_loss:.4f}")
        print(f"{'='*80}\n")


class AcademicTextDataset(Dataset):
    """Dataset cho việc fine-tune model tóm tắt văn bản học thuật"""
    
    def __init__(
        self,
        tokenizer,
        data_file: str,
        max_input_length: int = 512,
        max_target_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.examples = []
        
        self._load_data(data_file)
    
    def _load_data(self, data_file: str):
        """Load dữ liệu từ file .jsonl"""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File không tồn tại: {data_file}")
        
        logger.info(f"Đang tải dữ liệu từ: {data_file}")
        count_skipped = 0
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Kiểm tra dữ liệu cần thiết
                    if "full_text" not in data or "abstract" not in data:
                        count_skipped += 1
                        continue
                    
                    full_text = data["full_text"]
                    abstract = data["abstract"]
                    
                    # Xử lý NaN, None hoặc kiểu không phải string
                    if full_text is None or abstract is None:
                        count_skipped += 1
                        continue
                    
                    if not isinstance(full_text, str):
                        full_text = str(full_text)
                    if not isinstance(abstract, str):
                        abstract = str(abstract)
                    
                    full_text = full_text.strip()
                    abstract = abstract.strip()
                    
                    # Bỏ qua dữ liệu trống hoặc quá ngắn
                    if not full_text or not abstract or len(full_text) < 10:
                        count_skipped += 1
                        continue
                    
                    self.examples.append({
                        "input_text": full_text,
                        "target_text": abstract
                    })
                    
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    count_skipped += 1
                    continue
        
        logger.info(f"✅ Đã tải {len(self.examples)} ví dụ (bỏ qua {count_skipped})")
        
        if len(self.examples) == 0:
            raise ValueError("❌ Không tìm thấy dữ liệu hợp lệ trong file")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Encode input
        inputs = self.tokenizer(
            example["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"  # Trả về numpy array thay vì pt
        )
        
        # Encode target
        targets = self.tokenizer(
            example["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"  # Trả về numpy array thay vì pt
        )
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": targets["input_ids"][0],
            "decoder_attention_mask": targets["attention_mask"][0],
        }


class AcademicFineTuner:
    """Class quản lý việc fine-tune model tóm tắt văn bản học thuật"""
    
    def __init__(self, base_model_path: Optional[str] = None):
        self.fine_tuned_model_path = "./models/hhlai_academic_textsum"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"🔧 Sử dụng device: {self.device}")
        logger.info(f"💾 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Chọn base model
        self.base_model_path = base_model_path or self._select_base_model()
        self._load_base_model()
    
    def _select_base_model(self) -> str:
        models_dir = "./models"
        available_models = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
            and d != "hhlai_academic_textsum"
        ]
        
        if not available_models:
            raise FileNotFoundError("Không tìm thấy model nào trong ./models/")
        
        print("\n📦 Danh sách các model có sẵn:")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
        
        while True:
            try:
                choice = int(input(f"\nChọn model (1-{len(available_models)}): "))
                if 1 <= choice <= len(available_models):
                    selected = available_models[choice - 1]
                    return os.path.join(models_dir, selected)
            except ValueError:
                pass
            print("❌ Lựa chọn không hợp lệ")
    
    def _load_base_model(self):
        logger.info(f"📂 Đang tải base model từ: {self.base_model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_path,
                device_map='auto'
            )
            
            # Enable gradient checkpointing để tiết kiệm memory
            self.model.gradient_checkpointing_enable()
            
            logger.info("✅ Base model đã load thành công")
        except Exception as e:
            logger.error(f"❌ Lỗi khi load model: {e}")
            raise
    
    def fine_tune(
        self,
        data_file: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_input_length: int = 512,
        max_target_length: int = 256,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 2,
    ):
        """
        Giải thích thông số Fine-tune model
            data_file: Đường dẫn file dữ liệu .jsonl
            epochs: Số epoch huấn luyện
            batch_size: Batch size
            learning_rate: Learning rate
            max_input_length: Độ dài tối đa input
            max_target_length: Độ dài tối đa target
            warmup_steps: Số warmup steps
            weight_decay: Weight decay
            gradient_accumulation_steps: Số bước tích lũy gradient (tăng hiệu quả batch size)
        """
        logger.info("🚀 Bắt đầu fine-tune")
        
        # Tạo thư mục output
        os.makedirs(self.fine_tuned_model_path, exist_ok=True)
        
        # Load dataset
        logger.info("📊 Đang chuẩn bị dữ liệu...")
        dataset = AcademicTextDataset(
            tokenizer=self.tokenizer,
            data_file=data_file,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        
        # Hiển thị thông tin training
        total_steps = (len(dataset) // (batch_size * gradient_accumulation_steps)) * epochs
        print(f"\n📈 THÔNG TIN HUẤN LUYỆN:")
        print("=" * 70)
        print(f"   📝 Số lượng ví dụ: {len(dataset)}")
        print(f"   🔄 Số epoch: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   📦 Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   📊 Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"   📊 Tổng steps: {total_steps}")
        print(f"   🎯 Learning rate: {learning_rate}")
        print(f"   🌡️  Device: {self.device}")
        print(f"   💾 Output: {self.fine_tuned_model_path}")
        print("=" * 70)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
        )
        
        # Training arguments - TỐI ƯU CHO WINDOWS RTX 3050 Ti
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.fine_tuned_model_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            # FP32 - ổn định nhất cho Windows
            fp16=False,
            bf16=False,
            # Saving
            save_total_limit=1,
            save_steps=1000,
            # Logging - giảm overhead
            logging_steps=10,
            logging_first_step=False,
            logging_nan_inf_filter=False,
            # Optimization
            optim="adamw_torch",
            eval_strategy="no",
            # Memory optimization
            gradient_checkpointing=True,
            # Data loading (WINDOWS KHÔNG HỖ TRỢ NUM_WORKERS)
            dataloader_num_workers=0,  # BẮT BUỘC = 0 cho Windows
            dataloader_pin_memory=False,  # Tắt trên Windows
            # Report
            report_to=[],
            # Seed
            seed=42,
        )
        
        # Trainer với Progress và NaN callback
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[ProgressAndNaNCallback()],
        )
        
        # Fine-tune
        print("\n⏳ Đang huấn luyện...\n")
        start_time = time.time()
        
        try:
            trainer.train()
            
            training_time = time.time() - start_time
            training_time_str = str(timedelta(seconds=int(training_time)))
            
            print(f"\n\n✅ HUẤN LUYỆN HOÀN THÀNH!")
            print("=" * 70)
            print(f"   ⏱️  Thời gian: {training_time_str}")
            print(f"   📊 Tổng steps: {trainer.state.global_step}")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã dừng huấn luyện bởi người dùng")
            training_time = time.time() - start_time
            training_time_str = str(timedelta(seconds=int(training_time)))
            print(f"   ⏱️  Thời gian chạy: {training_time_str}")
        
        # Lưu model
        logger.info(f"💾 Đang lưu model vào: {self.fine_tuned_model_path}")
        self.model.save_pretrained(self.fine_tuned_model_path)
        self.tokenizer.save_pretrained(self.fine_tuned_model_path)
        
        print(f"\n✅ FINE-TUNE HOÀN THÀNH!")
        print(f"📂 Model đã được lưu tại: {self.fine_tuned_model_path}")
        
        # Kiểm tra kích thước model
        model_size = sum(
            os.path.getsize(os.path.join(self.fine_tuned_model_path, f))
            for f in os.listdir(self.fine_tuned_model_path)
            if os.path.isfile(os.path.join(self.fine_tuned_model_path, f))
        ) / (1024 * 1024)
        print(f"💾 Kích thước model: {model_size:.1f}MB\n")


def main():
    # Phân tích đối số CLI
    parser = argparse.ArgumentParser(
        description="Fine-tune model tóm tắt văn bản học thuật"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        help="Đường dẫn đến base model (VD: ./models/google_mt5-base)"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        default="./_crawler/datasets.jsonl",
        help="Đường dẫn đến file dữ liệu .jsonl"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Số epoch huấn luyện"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Độ dài tối đa input"
    )
    
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=256,
        help="Độ dài tối đa target"
    )
    
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Số warmup steps"
    )
    
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
        help="Gradient accumulation steps"
    )
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo fine-tuner
        fine_tuner = AcademicFineTuner(base_model_path=args.base_model)
        
        # Xác nhận file dữ liệu
        if not os.path.exists(args.data_file):
            logger.error(f"❌ File dữ liệu không tìm thấy: {args.data_file}")
            return
        
        # Xác nhận trước khi fine-tune
        print("\n" + "=" * 70)
        print("⚠️  XÁC NHẬN CẤU HÌNH:")
        print("=" * 70)
        print(f"📂 Base model: {fine_tuner.base_model_path}")
        print(f"📝 File dữ liệu: {args.data_file}")
        print(f"🎯 Output model: {fine_tuner.fine_tuned_model_path}")
        print("=" * 70)
        
        confirm = input("\n✅ Bạn có chắc muốn tiếp tục? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Đã hủy")
            return
        
        # Fine-tune
        fine_tuner.fine_tune(
            data_file=args.data_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation,
        )
    
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã dừng bởi người dùng")
        return
    
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    