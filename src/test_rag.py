import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic   # ← bu satır zaten var

load_dotenv()

def run_test():
    print("🔍 RAG Sistemi Test Ediliyor...")
    
    # ── GÜNCEL AYARLAR ──
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # ←←← BU SATIRI DEĞİŞTİR (en dengeli ve güncel model) ←←←
    Settings.llm = Anthropic(
        model="claude-sonnet-4-6",      # ← 2026’nın en iyi dengeli modeli (Sennet 4.6)
        temperature=0.0
    )
    # Alternatifler (istersen değiştirebilirsin):
    # model="claude-opus-4-6"     # en güçlü (biraz daha yavaş/pahalı)
    # model="claude-haiku-4-5"    # en hızlı ve ucuz (eski Haiku’nun yerini aldı)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    storage_path = os.path.join(project_root, "storage_native")
    print(f"📂 Yerel veritabanı yükleniyor: {storage_path}")
   
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Test sorguları (istediğin gibi genişletebilirsin)
    print("\n" + "="*60)
    print("TEST SORGUSU 1: Hata kodu 16#8180 ne anlama geliyor?")
    print("="*60)
    response1 = query_engine.query("What does the error code 16#8180 mean in Siemens systems? Explain briefly.")
    print(f"🤖 CLAUDE'UN CEVABI:\n{response1}\n")

    print("="*60)
    print("TEST SORGUSU 2: CPU STOP durumunda ne yapılmalı?")
    print("="*60)
    response2 = query_engine.query("What are the recommended actions or diagnostics when the CPU goes into STOP mode?")
    print(f"🤖 CLAUDE'UN CEVABI:\n{response2}\n")

if __name__ == "__main__":
    run_test()