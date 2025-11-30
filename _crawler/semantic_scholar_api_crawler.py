"""
Semantic Scholar API Crawler - KHÔNG BỊ CHẶN
API miễn phí, không cần key cho usage thấp
Docs: https://api.semanticscholar.org/
"""
import requests
import pandas as pd
import json
import os
import time
import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# CẤU HÌNH
# -------------------------------
TOPICS = [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "computer vision",
    "natural language processing",
    "blockchain",
    "IoT",
    "biotechnology",
    "renewable energy",
    "climate change"
]

PAPERS_PER_TOPIC = 200
TIMEOUT = 45
MAX_WORKERS = 6
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CSV = os.path.join(OUTPUT_DIR, "semantic_scholar_dataset.csv")
DATASET_JSONL = os.path.join(OUTPUT_DIR, "semantic_scholar_dataset.jsonl")
CRAWLED_IDS_FILE = os.path.join(OUTPUT_DIR, "semantic_scholar_crawled_ids.txt")
CRAWL_LOG_FILE = os.path.join(OUTPUT_DIR, "semantic_scholar_crawl_history.json")

# Semantic Scholar API
API_BASE = "https://api.semanticscholar.org/graph/v1"

# -------------------------------
# QUẢN LÝ IDs
# -------------------------------
def load_crawled_ids():
    if os.path.exists(CRAWLED_IDS_FILE):
        with open(CRAWLED_IDS_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_crawled_ids(ids):
    with open(CRAWLED_IDS_FILE, 'a', encoding='utf-8') as f:
        for paper_id in ids:
            f.write(f"{paper_id}\n")

def log_crawl_session(session_info):
    history = []
    if os.path.exists(CRAWL_LOG_FILE):
        with open(CRAWL_LOG_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append(session_info)
    with open(CRAWL_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# -------------------------------
# CRAWL SEMANTIC SCHOLAR API
# -------------------------------
def search_semantic_scholar(query, limit=10):
    """
    Search papers qua Semantic Scholar API
    """
    url = f"{API_BASE}/paper/search"
    params = {
        'query': query,
        'limit': limit,
        'fields': 'paperId,title,abstract,authors,year,venue,openAccessPdf,citationCount,fieldsOfStudy'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        papers = []
        for paper in data.get('data', []):
            try:
                paper_id = paper.get('paperId', '')
                if not paper_id:
                    continue
                
                # Extract authors
                authors = []
                for author in paper.get('authors', []):
                    if 'name' in author:
                        authors.append(author['name'])
                
                # Extract PDF URL
                pdf_url = ''
                open_access = paper.get('openAccessPdf')
                if open_access and isinstance(open_access, dict):
                    pdf_url = open_access.get('url', '')
                
                # Categories
                fields = paper.get('fieldsOfStudy', [])
                categories = ', '.join(fields) if fields else ''
                
                papers.append({
                    'id': f"S2-{paper_id}",
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'categories': categories,
                    'pdf_url': pdf_url,
                    'authors': ', '.join(authors),
                    'year': paper.get('year', ''),
                    'venue': paper.get('venue', ''),
                    'citations': paper.get('citationCount', 0)
                })
                
            except:
                continue
        
        return papers
        
    except Exception as e:
        print(f"  ❌ API Error: {e}")
        return []

def get_paper_details(paper_id):
    """
    Lấy chi tiết paper (nếu cần thêm thông tin)
    """
    url = f"{API_BASE}/paper/{paper_id}"
    params = {
        'fields': 'title,abstract,authors,year,venue,openAccessPdf,references,citations'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except:
        return None

# -------------------------------
# EXTRACT PDF
# -------------------------------
def extract_pdf_content(pdf_url):
    if not pdf_url:
        return ""
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, timeout=TIMEOUT, stream=True, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower():
            return ""
        
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text("text")
        doc.close()
        return full_text.strip()
    except:
        return ""

# -------------------------------
# PROCESS PAPER
# -------------------------------
def process_paper(paper_info):
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
# MAIN
# -------------------------------
def main():
    print(f"🚀 Semantic Scholar API Crawler")
    print(f"✅ Using official API - NO CAPTCHA, NO BAN")
    print(f"📚 {len(TOPICS)} topics × {PAPERS_PER_TOPIC} papers | {MAX_WORKERS} workers\n")
    
    crawled_ids = load_crawled_ids()
    print(f"📋 Database: {len(crawled_ids)} papers\n")
    
    start_time = time.time()
    all_metadata = []
    
    # ===== CRAWL METADATA =====
    print("📡 Crawling from Semantic Scholar API:")
    for topic_idx, topic in enumerate(TOPICS, 1):
        papers = search_semantic_scholar(topic, limit=20)
        
        # Lọc bài mới
        new_papers = [p for p in papers if p['id'] not in crawled_ids][:PAPERS_PER_TOPIC]
        
        for paper in new_papers:
            paper['topic'] = topic
        
        all_metadata.extend(new_papers)
        print(f"  [{topic_idx}/{len(TOPICS)}] {topic}: {len(new_papers)}")
        
        # Rate limiting nhẹ (API cho phép 100 req/5min)
        time.sleep(1)
    
    print(f"\n✅ {len(all_metadata)} new papers\n")
    
    if not all_metadata:
        print("⚠️  No new papers found.")
        return
    
    # ===== DOWNLOAD & EXTRACT =====
    print(f"📥 Extracting PDFs (only open access papers)...")
    papers_to_process = [(i+1, p) for i, p in enumerate(all_metadata)]
    all_papers = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_paper, p): p for p in papers_to_process}
        completed = 0
        
        for future in as_completed(futures):
            try:
                all_papers.append(future.result())
                completed += 1
                if completed % 5 == 0 or completed == len(papers_to_process):
                    print(f"  {completed}/{len(papers_to_process)} ({100*completed//len(papers_to_process)}%)")
            except:
                completed += 1
    
    all_papers.sort(key=lambda x: x["stt"])
    # Chấp nhận papers có abstract
    valid_papers = [p for p in all_papers if p["abstract"] and len(p["abstract"]) > 50]
    papers_with_fulltext = [p for p in valid_papers if p["full_text"] and len(p["full_text"]) > 500]
    
    print(f"✅ {len(valid_papers)} with abstract | {len(papers_with_fulltext)} with full text\n")
    
    # ===== SAVE =====
    csv_path, jsonl_path = append_to_dataset(valid_papers)
    
    if not csv_path or not jsonl_path:
        print("❌ Error saving dataset")
        return
    
    new_ids = [p['id'] for p in valid_papers]
    save_crawled_ids(new_ids)
    
    # ===== LOG =====
    elapsed = time.time() - start_time
    log_crawl_session({
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 1),
        "new_papers": len(valid_papers),
        "papers_with_fulltext": len(papers_with_fulltext),
        "total_in_database": len(crawled_ids) + len(new_ids),
        "topics": TOPICS,
        "papers_per_topic": PAPERS_PER_TOPIC
    })
    
    # ===== SUMMARY =====
    print("="*50)
    print(f"✨ Done in {elapsed/60:.1f}m")
    print(f"📊 New: {len(valid_papers)} ({len(papers_with_fulltext)} with PDF)")
    print(f"💾 Total: {len(crawled_ids) + len(new_ids)}")
    if csv_path and jsonl_path:
        print(f"📁 {os.path.basename(csv_path)} | {os.path.basename(jsonl_path)}")
    print("="*50)

if __name__ == "__main__":
    main()