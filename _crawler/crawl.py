import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json
import os
import time
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# CẤU HÌNH
# -------------------------------
# Danh sách topics để crawl - ĐÃ MỞ RỘNG ĐA LĨNH VỰC
TOPICS = [
    # === AI/ML Core ===
    "machine learning",
    "deep learning",
    "neural networks",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "generative adversarial networks",
    "transformer models",
    "large language models",
    "diffusion models",
    "graph neural networks",
    "meta-learning",
    "few-shot learning",
    "transfer learning",
    "self-supervised learning",
    "federated learning",
    "explainable AI",
    "multimodal learning",
    
    # === Physics & Chemistry ===
    "quantum computing",
    "quantum machine learning",
    "astrophysics",
    "cosmology",
    "particle physics",
    "condensed matter physics",
    "materials science",
    "computational chemistry",
    "molecular dynamics"
]

PAPERS_PER_TOPIC = 3        # Số bài mỗi topic (Giảm xuống vì có nhiều topics)
BATCH_SIZE = 3              # Số bài mỗi lần query API (max 50)
TIMEOUT = 45                # Timeout download PDF (giây)
MAX_WORKERS = 10            # Số luồng download song song
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# File lưu trữ
DATASET_CSV = os.path.join(OUTPUT_DIR, "arxiv_dataset.csv")
DATASET_JSONL = os.path.join(OUTPUT_DIR, "arxiv_dataset.jsonl")
CRAWLED_IDS_FILE = os.path.join(OUTPUT_DIR, "crawled_ids.txt")
CRAWL_LOG_FILE = os.path.join(OUTPUT_DIR, "crawl_history.json")

# -------------------------------
# QUẢN LÝ IDs ĐÃ CRAWL
# -------------------------------
def load_crawled_ids():
    """Load danh sách ID đã crawl từ file"""
    if os.path.exists(CRAWLED_IDS_FILE):
        with open(CRAWLED_IDS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_crawled_ids(ids):
    """Lưu danh sách ID mới vào file"""
    with open(CRAWLED_IDS_FILE, 'a') as f:
        for paper_id in ids:
            f.write(f"{paper_id}\n")

def log_crawl_session(session_info):
    """Lưu thông tin session crawl"""
    history = []
    if os.path.exists(CRAWL_LOG_FILE):
        with open(CRAWL_LOG_FILE, 'r') as f:
            history = json.load(f)
    
    history.append(session_info)
    
    with open(CRAWL_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# -------------------------------
# CRAWL ARXIV METADATA
# -------------------------------
def crawl_arxiv(query, start=0, max_results=50):
    """Crawl metadata từ arXiv API"""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",  # Sắp xếp theo ngày để ổn định hơn
        "sortOrder": "descending"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            try:
                pid_full = entry.find("atom:id", ns).text
                pid = pid_full.split("/abs/")[-1]
                
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                
                categories = []
                for category in entry.findall("atom:category", ns):
                    term = category.get("term")
                    if term:
                        categories.append(term)
                
                pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"
                
                papers.append({
                    "id": pid,
                    "title": title,
                    "abstract": summary,
                    "categories": ", ".join(categories),
                    "pdf_url": pdf_url
                })
                
            except Exception:
                continue

        return papers
    
    except Exception:
        return []


# -------------------------------
# EXTRACT PDF CONTENT (OPTIMIZED)
# -------------------------------
def extract_pdf_content(pdf_url):
    """Tải và trích xuất nội dung PDF (tối ưu)"""
    try:
        response = requests.get(pdf_url, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        pdf_bytes = response.content
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Trích xuất text nhanh hơn bằng cách không format
        full_text = ""
        for page in doc:
            full_text += page.get_text("text")
        
        doc.close()
        return full_text.strip()
    
    except:
        return ""


# -------------------------------
# PROCESS SINGLE PAPER
# -------------------------------
def process_paper(paper_info):
    """Xử lý 1 bài báo (download + extract)"""
    stt, paper = paper_info
    
    full_text = extract_pdf_content(paper["pdf_url"])
    
    return {
        "stt": stt,
        "id": paper["id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "full_text": full_text,
        "categories": paper["categories"],
        "topic": paper.get("topic", "")
    }


# -------------------------------
# APPEND DATASET
# -------------------------------
def append_to_dataset(new_data):
    """Thêm dữ liệu mới vào dataset hiện có"""
    if not new_data:
        return None, None
    
    existing_data = []
    if os.path.exists(DATASET_CSV):
        df_old = pd.read_csv(DATASET_CSV)
        existing_data = df_old.to_dict('records')
    
    all_data = existing_data + new_data
    for i, row in enumerate(all_data, 1):
        row['stt'] = i
    
    df = pd.DataFrame(all_data)
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
    
    with open(DATASET_JSONL, "w", encoding="utf-8") as f:
        for row in all_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    return DATASET_CSV, DATASET_JSONL


# -------------------------------
# MAIN PIPELINE (PARALLEL + MULTI-TOPIC + RESUME)
# -------------------------------
def main():
    print(f"🚀 ArXiv Crawler | {len(TOPICS)} topics × {PAPERS_PER_TOPIC} papers | {MAX_WORKERS} workers")
    
    crawled_ids = load_crawled_ids()
    print(f"📋 Database: {len(crawled_ids)} bài | Target: {len(TOPICS) * PAPERS_PER_TOPIC} bài mới\n")
    
    start_time = time.time()
    all_metadata = []
    
    # ===== CRAWL METADATA =====
    print("📡 Crawling metadata:")
    for topic_idx, topic in enumerate(TOPICS, 1):
        topic_papers = []
        crawl_start = 0
        
        while len(topic_papers) < PAPERS_PER_TOPIC and crawl_start < 500:
            papers = crawl_arxiv(topic, start=crawl_start, max_results=BATCH_SIZE)
            if not papers:
                break
            
            new_papers = [p for p in papers if p['id'] not in crawled_ids]
            topic_papers.extend(new_papers)
            crawl_start += BATCH_SIZE
            
            if len(topic_papers) >= PAPERS_PER_TOPIC:
                topic_papers = topic_papers[:PAPERS_PER_TOPIC]
                break
        
        for paper in topic_papers:
            paper['topic'] = topic
        
        all_metadata.extend(topic_papers)
        print(f"  [{topic_idx}/{len(TOPICS)}] {topic}: {len(topic_papers)}")
    
    print(f"✅ {len(all_metadata)} bài mới\n")
    
    if not all_metadata:
        print("⚠️  Không có bài mới. Tăng PAPERS_PER_TOPIC hoặc thử topic khác.")
        return
    
    # ===== DOWNLOAD & EXTRACT =====
    print(f"📥 Extracting PDFs...")
    papers_to_process = [(i+1, p) for i, p in enumerate(all_metadata)]
    all_papers = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_paper, p): p for p in papers_to_process}
        completed = 0
        
        for future in as_completed(futures):
            try:
                all_papers.append(future.result())
                completed += 1
                if completed % 50 == 0 or completed == len(papers_to_process):
                    print(f"  {completed}/{len(papers_to_process)} ({100*completed//len(papers_to_process)}%)")
            except:
                completed += 1
    
    all_papers.sort(key=lambda x: x["stt"])
    valid_papers = [p for p in all_papers if p["full_text"] and len(p["full_text"]) > 500]
    print(f"✅ {len(valid_papers)}/{len(all_papers)} success\n")
    
    # ===== SAVE =====
    csv_path, jsonl_path = append_to_dataset(valid_papers)
    
    # Kiểm tra nếu lưu thành công
    if not csv_path or not jsonl_path:
        print("❌ Lỗi khi lưu dataset")
        return
    
    new_ids = [p['id'] for p in valid_papers]
    save_crawled_ids(new_ids)
    
    # ===== LOG =====
    elapsed = time.time() - start_time
    log_crawl_session({
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 1),
        "new_papers": len(valid_papers),
        "total_in_database": len(crawled_ids) + len(new_ids),
        "topics": TOPICS,
        "papers_per_topic": PAPERS_PER_TOPIC
    })
    
    # ===== SUMMARY =====
    print("="*50)
    print(f"✨ Done in {elapsed/60:.1f}m")
    print(f"📊 New: {len(valid_papers)} | Total: {len(crawled_ids) + len(new_ids)}")
    if csv_path and jsonl_path:
        print(f"💾 {os.path.basename(csv_path)} | {os.path.basename(jsonl_path)}")
    print("="*50)


if __name__ == "__main__":
    main()
    