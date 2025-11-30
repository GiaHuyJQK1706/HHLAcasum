import json
import re
from pathlib import Path

# Cấu hình
MAX_INPUT_LEN = 16384
MIN_TEXT_LEN = 20  # bỏ full_text quá ngắn
MIN_ABSTRACT_LEN = 10  # bỏ abstract quá ngắn

def clean_text(text: str) -> str:
    """Làm sạch text: loại bỏ ký tự lạ, gộp khoảng trắng"""
    if not text:
        return ""
    text = str(text)
    text = text.replace("\f", " ").replace("\n", " ")
    # giữ ký tự in được + Vietnamese cơ bản
    text = re.sub(r"[^A-Za-z0-9À-ỹ.,;:!?()\"'%\-–—/ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def truncate_text(text: str, max_len: int) -> str:
    return text[:max_len] if text else ""

def clean_jsonl_file(input_file: str, output_file: str):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"❌ File không tồn tại: {input_file}")
        return

    cleaned_entries = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            full_text = truncate_text(clean_text(entry.get("full_text") or entry.get("article")), MAX_INPUT_LEN)
            abstract = truncate_text(clean_text(entry.get("abstract")), MAX_INPUT_LEN // 4)

            if len(full_text) < MIN_TEXT_LEN or len(abstract) < MIN_ABSTRACT_LEN:
                continue

            # abstract trước, full_text sau
            cleaned_entries.append({
                "abstract": abstract,
                "full_text": full_text
            })

    with output_path.open("w", encoding="utf-8") as f:
        for entry in cleaned_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ {input_file}: clean xong {len(cleaned_entries)} entries, lưu vào {output_file}")

if __name__ == "__main__":
    input_file = "arxiv_summarization_20k.jsonl"
    output_file = "arxiv_summarization_20k_clean.jsonl"
    clean_jsonl_file(input_file, output_file)
