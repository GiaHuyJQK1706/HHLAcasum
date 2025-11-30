import json

# Danh sách các file cần ghép
input_files = [
    "arxiv_datasets_clean.jsonl",
    "arxiv_summarization_20k_clean.jsonl",
    "semantic_scholar_datasets_clean.jsonl"
]

# File output cuối
output_file = "datasets.jsonl"

with open(output_file, "w", encoding="utf-8") as fout:
    for f in input_files:
        print(f"Đang ghép: {f}")
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                # Ghi nguyên dòng, không sửa nội dung
                fout.write(line)

print("🔥 Ghép xong toàn bộ 3 file vào datasets.jsonl!")