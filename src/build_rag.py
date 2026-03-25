import os
import sys
import time
import fitz  # PyMuPDF'in standart ve en garantili Python adıdır

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter

# ──────────────────────────────────────────────
# AYARLAR (SADECE GEREKLİ OLANLAR)
# ──────────────────────────────────────────────
sys.setrecursionlimit(2000)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=100)
# DİKKAT: LLM ve Tokenizer ayarları burada YOK çünkü okuma aşamasında LLM'e ihtiyaç yoktur!

STORAGE_DIR = "./storage_native"

def log(emoji, msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {emoji}  {msg}", flush=True)

def elapsed(start):
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}dk"

log("🧹", "Eski veritabanı (varsa) yok sayılıyor, PDF'ler temiz metin olarak baştan okunuyor...")

# ──────────────────────────────────────────
# ADIM 1: PDF OKUMA (Garantili Fitz Metodu)
# ──────────────────────────────────────────
log("📂", "docs/ klasöründeki PDF'ler PyMuPDF(fitz) ile DOĞRUDAN okunuyor...")
t0 = time.time()

pdf_files = [os.path.join("docs", f) for f in os.listdir("docs") if f.endswith('.pdf')]
documents = []

for file_path in pdf_files:
    log("📄", f"Okunuyor: {file_path}")
    try:
        # LlamaIndex'e güvenmek yerine PDF'i kendi ellerimizle açıyoruz
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        
        # Sadece saf metni LlamaIndex'in cebine koyuyoruz
        documents.append(Document(text=full_text, metadata={"file_name": os.path.basename(file_path)}))
    except Exception as e:
        log("❌", f"{file_path} okunamadı: {e}")

log("✅", f"PDF okuma tamamlandı ({elapsed(t0)}). Toplam doküman: {len(documents)}")

if len(documents) == 0:
    log("❌", "Hiç metin çıkarılamadı! Yolu kontrol et.")
    sys.exit(1)

log("🔄", "Embedding başlıyor... (Saf metin olduğu için çok hızlı sürecek)")

# ──────────────────────────────────────────
# ADIM 2: EMBEDDING + INDEX OLUŞTURMA
# ──────────────────────────────────────────
t1 = time.time()
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,
)
log("✅", f"Embedding tamamlandı ({elapsed(t1)}). Toplam chunk: {len(index.index_struct.nodes_dict)}")

# ──────────────────────────────────────────
# ADIM 3: DİSKE KAYDET
# ──────────────────────────────────────────
os.makedirs(STORAGE_DIR, exist_ok=True)
t2 = time.time()
index.storage_context.persist(persist_dir=STORAGE_DIR)
log("💾", f"Yeni ve temiz Index '{STORAGE_DIR}' klasörüne kaydedildi ({elapsed(t2)})")

log("🎉", "build_rag.py KUSURSUZCA tamamlandı! Artık testi çalıştırabilirsin.")