import json

input_file = "semantic_scholar_dataset_clean.jsonl"
output_file = "semantic_scholar_datasets_clean.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        obj = json.loads(line)

        # Chỉ giữ 2 trường cần thiết (nếu không tồn tại thì đặt rỗng)
        new_obj = {
            "abstract": obj.get("abstract", ""),
            "full_text": obj.get("full_text", "")
        }

        fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

print("Done! Xuất file xong rồi đó 😎")
